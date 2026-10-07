from app.algorithms.allocation import GPUResourceManager
from app.algorithms.deadlock import DeadlockDetector
from app.algorithms.gpu_selection import GPUSelector
from datetime import datetime, timezone


class Simulation:

    # ==========================================
    # SUPPORTED SCHEDULING POLICIES
    # ==========================================

    SUPPORTED_POLICIES = [
        "FCFS",
        "PRIORITY",
        "SJF",
        "RESOURCE_EFFICIENT",
        "RESOURCE_BALANCED",
        "PREEMPTIVE"
    ]

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(self):

        # ==========================================
        # MULTI-GPU CLUSTER
        # ==========================================

        self.gpu_selector = GPUSelector(
            gpu_count=3
        )

        # Keep GPU-0 available through the legacy
        # self.gpu reference for compatibility.
        self.gpu = self.gpu_selector.get_gpu(
            "GPU-0"
        )

        self.jobs = []
        self.deadlock_detector = DeadlockDetector()
        self.job_sequence = 0
        self.scheduling_policy = "RESOURCE_BALANCED"
        self.preemption_count = 0
        self.preempted_job_ids = []

        # ==========================================
        # HISTORICAL ANALYTICS
        # ==========================================

        self.analytics_started_at = datetime.now(timezone.utc)
        self.completed_job_history = []
        self.cancelled_job_history = []
        self.allocation_attempts = 0
        self.successful_allocations = 0
        self.total_preemptions = 0

        # ==========================================
        # WORKLOAD / STRESS TEST HISTORY
        # ==========================================

        self.workload_batches = []
        self.stress_test_runs = []

    # ==========================================
    # JOB MANAGEMENT
    # ==========================================

    def add_job(self, job):
        self.job_sequence += 1
        job.queue_position = self.job_sequence
        self.jobs.append(job)

    def get_jobs(self):
        return self.jobs

    # ==========================================
    # RESOURCE REQUEST
    # ==========================================

    def get_job_resource_request(self, job):
        return {
            "compute_units": job.compute_required,
            "memory": job.memory_required,
            "accelerator_a": job.accelerator_a_required,
            "accelerator_b": job.accelerator_b_required
        }

    # ==========================================
    # GPU PLACEMENT HELPERS
    # ==========================================

    def get_job_gpu_id(self, job):

        return getattr(
            job,
            "gpu_id",
            None
        )

    def set_job_gpu_id(self, job, gpu_id):

        job.gpu_id = gpu_id

    def can_allocate_on_any_gpu(self, request):

        return bool(
            self.gpu_selector.get_candidate_gpus(
                request
            )
        )

    def select_best_gpu(self, request):

        return self.gpu_selector.select_best_gpu(
            request
        )

    def allocate_on_best_gpu(self, job):

        request = self.get_job_resource_request(job)

        selection = self.select_best_gpu(
            request
        )

        if selection is None:

            return None

        gpu_id = selection["gpu_id"]
        gpu = self.gpu_selector.get_gpu(
            gpu_id
        )

        success = gpu.allocate(request)

        if not success:

            return None

        self.set_job_gpu_id(
            job,
            gpu_id
        )

        self.successful_allocations += 1

        return {
            "gpu_id": gpu_id,
            "score": selection["score"]
        }

    # ==========================================
    # ALLOCATION EXPLANATION
    # ==========================================

    def get_job_allocation_explanation(self, job):
        """
        Explain why the scheduler selected a GPU for a job.

        The explanation is derived from the same candidate GPU
        scores used by the allocator, so the UI can show the
        scheduling decision instead of presenting only the result.
        """

        request = self.get_job_resource_request(job)
        selected_gpu_id = self.get_job_gpu_id(job)

        try:
            candidate_ids = self.gpu_selector.get_gpu_ids()
        except Exception:
            candidate_ids = []

        candidates = []

        for gpu_id in candidate_ids:

            gpu = self.gpu_selector.get_gpu(gpu_id)

            if gpu is None:
                continue

            try:
                can_allocate = bool(
                    gpu.can_allocate(request)
                )
            except Exception:
                can_allocate = False

            score = 0.0

            if can_allocate:
                try:
                    score = round(
                        float(
                            gpu.get_allocation_score(request)
                        ),
                        2
                    )
                except Exception:
                    score = 0.0

            candidates.append({
                "gpu_id": gpu_id,
                "eligible": can_allocate,
                "score": score
            })

        eligible_candidates = [
            candidate
            for candidate in candidates
            if candidate["eligible"]
        ]

        ranked_candidates = sorted(
            eligible_candidates,
            key=lambda candidate: candidate["score"],
            reverse=True
        )

        if selected_gpu_id is None:

            return {
                "allocated": False,
                "gpu_id": None,
                "score": round(
                    float(
                        getattr(job, "allocation_score", 0.0)
                    ),
                    2
                ),
                "decision": "WAITING",
                "headline": "No GPU currently satisfies this job.",
                "reasons": [
                    "The requested resource combination cannot currently fit on an available GPU.",
                    "The job remains in the scheduler waiting queue.",
                    "The allocation score will be recalculated as resources become available."
                ],
                "candidates": candidates,
                "policy": self.scheduling_policy
            }

        selected_score = round(
            float(
                getattr(job, "allocation_score", 0.0)
            ),
            2
        )

        rank = next(
            (
                index + 1
                for index, candidate in enumerate(ranked_candidates)
                if candidate["gpu_id"] == selected_gpu_id
            ),
            None
        )

        reasons = [
            "All requested compute, memory, and accelerator resources fit on the selected GPU.",
            "The scheduler evaluated the available GPU candidates before allocation.",
            "The selected GPU received the highest allocation score among eligible candidates.",
            "The score considers resource balance, fragmentation impact, and resource efficiency."
        ]

        if rank == 1:
            reasons.append(
                "This GPU ranked #1 among the currently eligible GPU candidates."
            )

        elif rank is not None:
            reasons.append(
                f"This GPU ranked #{rank} among eligible candidates."
            )

        return {
            "allocated": True,
            "gpu_id": selected_gpu_id,
            "score": selected_score,
            "decision": "ALLOCATED",
            "headline": f"{selected_gpu_id} selected by the scheduler.",
            "reasons": reasons,
            "candidates": candidates,
            "policy": self.scheduling_policy
        }

    def release_job_resources(self, job):

        request = self.get_job_resource_request(job)
        gpu_id = self.get_job_gpu_id(job)

        if gpu_id is None:
            return False

        gpu = self.gpu_selector.get_gpu(
            gpu_id
        )

        if gpu is None:
            return False

        gpu.release(request)
        return True

    # ==========================================
    # ALLOCATION SCORE
    # ==========================================

    def calculate_job_allocation_score(self, job):

        request = self.get_job_resource_request(job)

        try:

            gpu_id = self.get_job_gpu_id(job)

            # A running job is evaluated on the GPU where
            # it is currently placed.
            if gpu_id is not None:

                gpu = self.gpu_selector.get_gpu(
                    gpu_id
                )

                if gpu is not None:
                    return round(
                        float(
                            gpu.get_allocation_score(
                                request
                            )
                        ),
                        2
                    )

            # Waiting/queued jobs are evaluated against
            # the best GPU currently available.
            selection = self.select_best_gpu(
                request
            )

            if selection is None:
                return 0.0

            return round(
                float(selection["score"]),
                2
            )

        except Exception:

            return 0.0

    # ==========================================
    # TRY ALLOCATING A JOB
    # ==========================================

    def try_allocate_job(self, job):

        request = self.get_job_resource_request(job)

        self.allocation_attempts += 1

        # Select the best GPU rather than always using GPU-0.
        allocation = self.allocate_on_best_gpu(
            job
        )

        if allocation is not None:

            job.allocation_score = round(
                float(allocation["score"]),
                2
            )

            job.start()
            return True

        # No GPU can currently satisfy the request.
        job.wait()

        job.allocation_score = (
            self.calculate_job_allocation_score(job)
        )

        if self.scheduling_policy == "PREEMPTIVE":
            return self.try_preempt_for_job(job)

        return False

    # ==========================================
    # COMPLETE JOB
    # ==========================================

    def complete_job(self, job):

        if job.is_running():
            self.release_job_resources(job)
            job.gpu_id = None

        job.complete()
        job.allocation_score = 0.0

        self.completed_job_history.append({
            "job_id": job.job_id,
            "completed_at": job.completed_at.isoformat()
                if job.completed_at else None,
            "queue_wait_seconds": job.get_queue_wait_seconds(),
            "waiting_duration_seconds": job.get_waiting_duration_seconds(),
            "execution_duration_seconds": job.get_execution_duration_seconds(),
            "total_duration_seconds": job.get_total_duration_seconds(),
            "gpu_id": self.get_job_gpu_id(job)
        })

        self.retry_waiting_jobs()
        return True

    # ==========================================
    # CANCEL JOB
    # ==========================================

    def cancel_job(self, job):

        if job.is_running():
            self.release_job_resources(job)
            job.gpu_id = None

        if (
            job.is_queued()
            or job.is_waiting()
            or job.is_running()
        ):

            job.cancel()
            job.allocation_score = 0.0

            self.cancelled_job_history.append({
                "job_id": job.job_id,
                "cancelled_at": job.cancelled_at.isoformat()
                    if job.cancelled_at else None,
                "queue_wait_seconds": job.get_queue_wait_seconds(),
                "waiting_duration_seconds": job.get_waiting_duration_seconds(),
                "execution_duration_seconds": job.get_execution_duration_seconds(),
                "total_duration_seconds": job.get_total_duration_seconds(),
                "gpu_id": self.get_job_gpu_id(job)
            })

            self.retry_waiting_jobs()
            return True

        return False

    # ==========================================
    # RESOURCE EFFICIENCY
    # ==========================================

    def calculate_resource_efficiency_for_gpu(
        self,
        job,
        gpu
    ):

        total_resources = gpu.get_total_resources()
        request = self.get_job_resource_request(job)

        normalized_footprints = []

        resource_pairs = [
            (
                request.get("compute_units", 0),
                total_resources.get("compute_units", 0)
            ),
            (
                request.get("memory", 0),
                total_resources.get("memory", 0)
            ),
            (
                request.get("accelerator_a", 0),
                total_resources.get("accelerator_a", 0)
            ),
            (
                request.get("accelerator_b", 0),
                total_resources.get("accelerator_b", 0)
            )
        ]

        for requested, capacity in resource_pairs:

            if capacity <= 0 or requested <= 0:
                continue

            footprint = requested / capacity

            footprint = max(
                0.0,
                min(footprint, 1.0)
            )

            normalized_footprints.append(
                footprint
            )

        if not normalized_footprints:
            return 100.0

        average_footprint = (
            sum(normalized_footprints)
            / len(normalized_footprints)
        )

        return round(
            (1.0 - average_footprint) * 100,
            2
        )

    def calculate_resource_efficiency(self, job):
        """
        Calculate resource efficiency against the individual
        GPU nodes in the cluster. A smaller normalized resource
        footprint produces a higher efficiency score.
        """

        gpu_ids = self.gpu_selector.get_gpu_ids()

        if not gpu_ids:
            return 0.0

        scores = []

        for gpu_id in gpu_ids:

            gpu = self.gpu_selector.get_gpu(
                gpu_id
            )

            if gpu is None:
                continue

            if not gpu.can_allocate(
                self.get_job_resource_request(job)
            ):
                continue

            scores.append(
                self.calculate_resource_efficiency_for_gpu(
                    job,
                    gpu
                )
            )

        if not scores:
            return 0.0

        return round(
            max(scores),
            2
        )

    # ==========================================
    # JOB SIZE
    # ==========================================

    def calculate_job_size(self, job):
        return (
            job.compute_required
            + job.memory_required
            + job.accelerator_a_required
            + job.accelerator_b_required
        )

    # ==========================================
    # POLICY SORTING
    # ==========================================

    def sort_jobs_by_policy(self, jobs):
        jobs = list(jobs)
        policy = self.scheduling_policy

        if policy == "FCFS":
            return sorted(
                jobs,
                key=lambda job: job.queue_position
            )

        if policy == "PRIORITY":
            return sorted(
                jobs,
                key=lambda job: (
                    -job.priority,
                    job.queue_position
                )
            )

        if policy == "SJF":
            return sorted(
                jobs,
                key=lambda job: (
                    self.calculate_job_size(job),
                    job.queue_position
                )
            )

        if policy == "RESOURCE_EFFICIENT":
            return sorted(
                jobs,
                key=lambda job: (
                    -self.calculate_resource_efficiency(job),
                    job.queue_position
                )
            )

        if policy == "RESOURCE_BALANCED":
            for job in jobs:
                request = self.get_job_resource_request(job)
                try:
                    if self.can_allocate_on_any_gpu(request):
                        job.allocation_score = self.calculate_job_allocation_score(job)
                except Exception:
                    pass

            return sorted(
                jobs,
                key=lambda job: (
                    -float(job.allocation_score),
                    -job.priority,
                    job.queue_position
                )
            )

        if policy == "PREEMPTIVE":
            return sorted(
                jobs,
                key=lambda job: (
                    -job.priority,
                    job.queue_position
                )
            )

        return sorted(
            jobs,
            key=lambda job: job.queue_position
        )

    # ==========================================
    # UPDATE WAITING JOB SCORES
    # ==========================================

    def update_waiting_job_scores(self):
        for job in self.jobs:
            if job.is_waiting():
                job.allocation_score = self.calculate_job_allocation_score(job)

    # ==========================================
    # PREEMPTION DECISION
    # ==========================================

    def should_preempt(self, waiting_job, running_job):
        if not waiting_job.is_waiting():
            return False

        if not running_job.is_running():
            return False

        policy = self.scheduling_policy

        if policy == "PREEMPTIVE":
            if waiting_job.priority > running_job.priority:
                return True

            if waiting_job.priority == running_job.priority:
                return (
                    waiting_job.queue_position
                    < running_job.queue_position
                )

            return False

        if policy == "PRIORITY":
            return waiting_job.priority > running_job.priority

        if policy == "SJF":
            return (
                self.calculate_job_size(waiting_job)
                < self.calculate_job_size(running_job)
            )

        if policy == "RESOURCE_EFFICIENT":
            return (
                self.calculate_resource_efficiency(waiting_job)
                > self.calculate_resource_efficiency(running_job)
            )

        if policy == "RESOURCE_BALANCED":
            waiting_score = self.calculate_job_allocation_score(waiting_job)
            running_score = self.calculate_job_allocation_score(running_job)

            return (
                waiting_score > running_score
                and waiting_job.priority >= running_job.priority
            )

        if policy == "FCFS":
            return waiting_job.queue_position < running_job.queue_position

        return False

    # ==========================================
    # SELECT PREEMPTION VICTIM
    # ==========================================

    def select_preemption_victim(self, waiting_job):
        running_jobs = self.get_running_jobs()

        if not running_jobs:
            return None

        eligible_jobs = [
            job
            for job in running_jobs
            if self.should_preempt(waiting_job, job)
        ]

        if not eligible_jobs:
            return None

        if self.scheduling_policy in (
            "PREEMPTIVE",
            "PRIORITY"
        ):
            return min(
                eligible_jobs,
                key=lambda job: (
                    job.priority,
                    -job.queue_position
                )
            )

        if self.scheduling_policy == "SJF":
            return max(
                eligible_jobs,
                key=lambda job: (
                    self.calculate_job_size(job),
                    -job.queue_position
                )
            )

        if self.scheduling_policy == "RESOURCE_EFFICIENT":
            return min(
                eligible_jobs,
                key=lambda job: (
                    self.calculate_resource_efficiency(job),
                    job.queue_position
                )
            )

        if self.scheduling_policy == "RESOURCE_BALANCED":
            return min(
                eligible_jobs,
                key=lambda job: (
                    self.calculate_job_allocation_score(job),
                    job.priority
                )
            )

        return max(
            eligible_jobs,
            key=lambda job: job.queue_position
        )

    # ==========================================
    # PREEMPT RUNNING JOB
    # ==========================================

    def preempt_job(self, job):

        if not job.is_running():
            return False

        gpu_id = self.get_job_gpu_id(job)

        if gpu_id is None:
            return False

        released = self.release_job_resources(
            job
        )

        if not released:
            return False

        job.wait()

        job.allocation_score = (
            self.calculate_job_allocation_score(job)
        )

        self.preemption_count += 1

        if job.job_id not in self.preempted_job_ids:
            self.preempted_job_ids.append(
                job.job_id
            )

        return True

    # ==========================================
    # TRY PREEMPTING FOR A WAITING JOB
    # ==========================================

    def try_preempt_for_job(self, waiting_job):

        if not waiting_job.is_waiting():
            return False

        request = self.get_job_resource_request(
            waiting_job
        )

        # First check whether a GPU is now available.
        allocation = self.allocate_on_best_gpu(
            waiting_job
        )

        if allocation is not None:

            waiting_job.allocation_score = round(
                float(allocation["score"]),
                2
            )

            waiting_job.start()
            return True

        victim = self.select_preemption_victim(
            waiting_job
        )

        if victim is None:
            return False

        victim_gpu_id = self.get_job_gpu_id(
            victim
        )

        victim_request = (
            self.get_job_resource_request(victim)
        )

        if not self.preempt_job(victim):
            return False

        # Try all GPUs again after releasing the victim.
        allocation = self.allocate_on_best_gpu(
            waiting_job
        )

        if allocation is not None:

            waiting_job.allocation_score = round(
                float(allocation["score"]),
                2
            )

            waiting_job.start()
            return True

        # The waiting job still cannot fit. Restore
        # the preempted job to its original GPU.
        if victim_gpu_id is not None:

            victim_gpu = self.gpu_selector.get_gpu(
                victim_gpu_id
            )

            if victim_gpu is not None:

                if victim_gpu.can_allocate(
                    victim_request
                ):

                    restored = victim_gpu.allocate(
                        victim_request
                    )

                    if restored:

                        victim.start()
                        victim.gpu_id = victim_gpu_id

                        if (
                            victim.job_id
                            in self.preempted_job_ids
                        ):
                            self.preempted_job_ids.remove(
                                victim.job_id
                            )

                        self.preemption_count = max(
                            0,
                            self.preemption_count - 1
                        )

        return False

    # ==========================================
    # PREEMPTION PASS
    # ==========================================

    def run_preemption_pass(self):
        if self.scheduling_policy != "PREEMPTIVE":
            return False

        waiting_jobs = [
            job
            for job in self.jobs
            if job.is_waiting()
        ]

        if not waiting_jobs:
            return False

        self.update_waiting_job_scores()

        ordered_jobs = self.sort_jobs_by_policy(waiting_jobs)
        preempted_any = False

        for job in ordered_jobs:
            if not job.is_waiting():
                continue

            if self.try_preempt_for_job(job):
                preempted_any = True

        return preempted_any

    # ==========================================
    # RETRY WAITING JOBS
    # ==========================================

    def retry_waiting_jobs(self):

        waiting_jobs = [
            job
            for job in self.jobs
            if job.is_waiting()
        ]

        if not waiting_jobs:
            return

        if self.scheduling_policy == "PREEMPTIVE":

            self.run_preemption_pass()

            waiting_jobs = [
                job
                for job in self.jobs
                if job.is_waiting()
            ]

            if not waiting_jobs:
                return

        self.update_waiting_job_scores()

        ordered_jobs = self.sort_jobs_by_policy(
            waiting_jobs
        )

        for job in ordered_jobs:

            if not job.is_waiting():
                continue

            allocation = self.allocate_on_best_gpu(
                job
            )

            if allocation is None:
                continue

            job.allocation_score = round(
                float(allocation["score"]),
                2
            )

            job.start()

        self.update_waiting_job_scores()

    # ==========================================
    # WAITING JOBS
    # ==========================================

    def get_waiting_jobs(self):
        waiting_jobs = [
            job
            for job in self.jobs
            if job.is_waiting()
        ]

        return self.sort_jobs_by_policy(waiting_jobs)

    # ==========================================
    # RUNNING JOBS
    # ==========================================

    def get_running_jobs(self):
        return [
            job
            for job in self.jobs
            if job.is_running()
        ]

    # ==========================================
    # COMPLETED JOBS
    # ==========================================

    def get_completed_jobs(self):
        return [
            job
            for job in self.jobs
            if job.is_completed()
        ]

    # ==========================================
    # CANCELLED JOBS
    # ==========================================

    def get_cancelled_jobs(self):
        return [
            job
            for job in self.jobs
            if job.is_cancelled()
        ]

    # ==========================================
    # DEADLOCK DETECTION
    # ==========================================

    def detect_deadlock(self):
        try:
            waiting_jobs = self.get_waiting_jobs()
            running_jobs = self.get_running_jobs()
            available_resources = self.get_available_resources()

            try:
                return bool(
                    self.deadlock_detector.detect_deadlock(
                        waiting_jobs,
                        running_jobs,
                        available_resources
                    )
                )
            except TypeError:
                pass

            try:
                return bool(
                    self.deadlock_detector.detect_deadlock(
                        self.jobs,
                        available_resources
                    )
                )
            except TypeError:
                pass

            try:
                return bool(
                    self.deadlock_detector.detect_deadlock(self.jobs)
                )
            except TypeError:
                return False

        except Exception:
            return False

    # ==========================================
    # DEADLOCK STATUS
    # ==========================================

    def get_deadlock_status(self):
        detected = self.detect_deadlock()

        return {
            "deadlock_detected": detected,
            "status": (
                "DEADLOCK DETECTED"
                if detected
                else "SAFE"
            )
        }

    # ==========================================
    # RESOURCE METHODS
    # ==========================================

    def get_total_resources(self):

        return self.gpu_selector.get_cluster_total_resources()

    def get_available_resources(self):

        return self.gpu_selector.get_cluster_available_resources()

    def get_utilization(self):

        total = self.get_total_resources()
        available = self.get_available_resources()

        utilization = {}

        for resource, total_amount in total.items():

            if total_amount <= 0:
                utilization[resource] = 0.0
                continue

            used = (
                total_amount
                - available.get(resource, 0)
            )

            utilization[resource] = round(
                (used / total_amount) * 100,
                2
            )

        return utilization

    def get_availability_percentage(self):

        total = self.get_total_resources()
        available = self.get_available_resources()
        result = {}

        for resource, total_amount in total.items():

            if total_amount <= 0:
                result[resource] = 0.0
                continue

            result[resource] = round(
                (
                    available.get(resource, 0)
                    / total_amount
                ) * 100,
                2
            )

        return result

    def get_fragmentation(self):

        utilization = self.get_utilization()

        if not utilization:
            return {}

        values = list(
            utilization.values()
        )

        average_utilization = (
            sum(values) / len(values)
        )

        return {
            resource: round(
                abs(
                    value - average_utilization
                ),
                2
            )
            for resource, value
            in utilization.items()
        }

    def get_overall_fragmentation(self):

        utilization = self.get_utilization()

        if not utilization:
            return 0.0

        values = list(
            utilization.values()
        )

        return round(
            min(
                max(values) - min(values),
                100
            ),
            2
        )

    def get_fragmentation_level(self):

        fragmentation = (
            self.get_overall_fragmentation()
        )

        if fragmentation <= 20:
            return "LOW"

        if fragmentation <= 50:
            return "MEDIUM"

        return "HIGH"

    def get_allocation_efficiency(self):

        utilization = self.get_utilization()

        if not utilization:
            return 100.0

        values = list(
            utilization.values()
        )

        average_utilization = (
            sum(values) / len(values)
        )

        fragmentation = (
            self.get_overall_fragmentation()
        )

        efficiency = (
            average_utilization
            - (fragmentation * 0.5)
        )

        return round(
            max(
                0.0,
                min(efficiency, 100.0)
            ),
            2
        )

    def get_resource_balance_score(self):

        fragmentation = (
            self.get_overall_fragmentation()
        )

        return round(
            max(
                0.0,
                min(100.0 - fragmentation, 100.0)
            ),
            2
        )

    def get_resource_analysis(self):

        analysis = {
            "total": self.get_total_resources(),
            "available": self.get_available_resources(),
            "utilization": self.get_utilization(),
            "availability": self.get_availability_percentage(),
            "fragmentation": self.get_fragmentation(),
            "overall_fragmentation": (
                self.get_overall_fragmentation()
            ),
            "fragmentation_level": (
                self.get_fragmentation_level()
            ),
            "allocation_efficiency": (
                self.get_allocation_efficiency()
            ),
            "resource_balance_score": (
                self.get_resource_balance_score()
            ),
            "gpu_count": len(
                self.gpu_selector.get_gpu_ids()
            ),
            "gpus": self.gpu_selector.get_gpu_analysis()
        }

        return analysis

    # ==========================================
    # SCHEDULING POLICY
    # ==========================================

    def get_scheduling_policy(self):
        return {
            "policy": self.scheduling_policy,
            "available_policies": self.SUPPORTED_POLICIES
        }

    def set_scheduling_policy(self, policy):
        if policy is None:
            return False

        normalized_policy = str(policy).strip().upper()

        if normalized_policy not in self.SUPPORTED_POLICIES:
            return False

        self.scheduling_policy = normalized_policy

        self.update_waiting_job_scores()
        self.retry_waiting_jobs()

        return True

    # ==========================================
    # PREEMPTION STATISTICS
    # ==========================================

    def get_preemption_statistics(self):
        return {
            "preemption_count": self.preemption_count,
            "preempted_job_ids": list(self.preempted_job_ids),
            "preemptive_scheduling": (
                self.scheduling_policy == "PREEMPTIVE"
            )
        }

    # ==========================================
    # WORKLOAD / STRESS TEST HISTORY
    # ==========================================

    def record_workload_batch(self, result):
        self.workload_batches.append(result)

        # Keep the in-memory simulation lightweight while still
        # retaining enough recent history for dashboard reporting.
        if len(self.workload_batches) > 20:
            self.workload_batches = self.workload_batches[-20:]

    def record_stress_test(self, result):
        self.stress_test_runs.append(result)

        if len(self.stress_test_runs) > 20:
            self.stress_test_runs = self.stress_test_runs[-20:]

    def get_workload_statistics(self):
        return {
            "workload_batches": len(self.workload_batches),
            "stress_test_runs": len(self.stress_test_runs),
            "recent_workload_batches": list(self.workload_batches[-10:]),
            "recent_stress_tests": list(self.stress_test_runs[-10:])
        }

    # ==========================================
    # HISTORICAL ANALYTICS
    # ==========================================

    def get_historical_analytics(self):

        now = datetime.now(timezone.utc)

        elapsed_seconds = max(
            0.0,
            (now - self.analytics_started_at).total_seconds()
        )

        completed_jobs = [
            job for job in self.jobs
            if job.is_completed()
        ]

        cancelled_jobs = [
            job for job in self.jobs
            if job.is_cancelled()
        ]

        finished_jobs = completed_jobs + cancelled_jobs

        if finished_jobs:
            average_waiting_time = round(
                sum(
                    job.get_queue_wait_seconds()
                    for job in finished_jobs
                ) / len(finished_jobs),
                2
            )

            average_execution_time = round(
                sum(
                    job.get_execution_duration_seconds()
                    for job in finished_jobs
                ) / len(finished_jobs),
                2
            )

            average_total_duration = round(
                sum(
                    job.get_total_duration_seconds()
                    for job in finished_jobs
                ) / len(finished_jobs),
                2
            )
        else:
            average_waiting_time = 0.0
            average_execution_time = 0.0
            average_total_duration = 0.0

        completed_count = len(completed_jobs)
        total_submitted = len(self.jobs)

        successful_allocation_rate = (
            round(
                (
                    self.successful_allocations
                    / self.allocation_attempts
                ) * 100,
                2
            )
            if self.allocation_attempts > 0
            else 0.0
        )

        throughput_per_minute = (
            round(
                completed_count
                / (elapsed_seconds / 60),
                2
            )
            if elapsed_seconds > 0
            else 0.0
        )

        throughput_per_hour = (
            round(
                completed_count
                / (elapsed_seconds / 3600),
                2
            )
            if elapsed_seconds > 0
            else 0.0
        )

        try:
            cluster_utilization = self.get_utilization()
        except Exception:
            cluster_utilization = {}

        try:
            allocation_efficiency = (
                self.get_allocation_efficiency()
            )
        except Exception:
            allocation_efficiency = 0.0

        try:
            fragmentation = (
                self.get_overall_fragmentation()
            )
        except Exception:
            fragmentation = 0.0

        return {
            "analytics_started_at":
                self.analytics_started_at.isoformat(),

            "analytics_elapsed_seconds":
                round(elapsed_seconds, 2),

            "total_submitted_jobs":
                total_submitted,

            "completed_jobs":
                completed_count,

            "cancelled_jobs":
                len(cancelled_jobs),

            "active_jobs":
                len([
                    job for job in self.jobs
                    if job.is_running() or job.is_waiting()
                ]),

            "average_waiting_time":
                average_waiting_time,

            "average_execution_time":
                average_execution_time,

            "average_total_duration":
                average_total_duration,

            "throughput_per_minute":
                throughput_per_minute,

            "throughput_per_hour":
                throughput_per_hour,

            "allocation_attempts":
                self.allocation_attempts,

            "successful_allocations":
                self.successful_allocations,

            "successful_allocation_rate":
                successful_allocation_rate,

            "allocation_efficiency":
                allocation_efficiency,

            "fragmentation":
                fragmentation,

            "cluster_utilization":
                cluster_utilization,

            "gpu_count":
                len(self.gpu_selector.get_gpu_ids()),

            "gpu_utilization":
                self.gpu_selector.get_gpu_utilization(),

            "gpu_availability":
                self.gpu_selector.get_gpu_availability(),

            "preemption_count":
                self.preemption_count,

            "completed_history":
                list(self.completed_job_history),

            "cancelled_history":
                list(self.cancelled_job_history)
        }

    # ==========================================
    # STATISTICS
    # ==========================================

    def get_statistics(self):
        queued_jobs = [
            job for job in self.jobs
            if job.is_queued()
        ]

        waiting_jobs = [
            job for job in self.jobs
            if job.is_waiting()
        ]

        running_jobs = [
            job for job in self.jobs
            if job.is_running()
        ]

        completed_jobs = [
            job for job in self.jobs
            if job.is_completed()
        ]

        cancelled_jobs = [
            job for job in self.jobs
            if job.is_cancelled()
        ]

        completed_or_running = [
            job
            for job in self.jobs
            if job.is_completed() or job.is_running()
        ]

        if completed_or_running:
            average_waiting_time = round(
                sum(
                    job.get_queue_wait_seconds()
                    for job in completed_or_running
                ) / len(completed_or_running),
                2
            )
        else:
            average_waiting_time = 0.0

        execution_jobs = [
            job
            for job in self.jobs
            if job.is_completed() or job.is_running()
        ]

        if execution_jobs:
            average_execution_time = round(
                sum(
                    job.get_execution_duration_seconds()
                    for job in execution_jobs
                ) / len(execution_jobs),
                2
            )
        else:
            average_execution_time = 0.0

        try:
            utilization = self.get_utilization()
        except Exception:
            utilization = {}

        try:
            allocation_efficiency = self.get_allocation_efficiency()
        except Exception:
            allocation_efficiency = 0.0

        try:
            fragmentation = self.get_overall_fragmentation()
        except Exception:
            fragmentation = 0.0

        return {
            "total_jobs": len(self.jobs),
            "queued_jobs": len(queued_jobs),
            "waiting_jobs": len(waiting_jobs),
            "running_jobs": len(running_jobs),
            "completed_jobs": len(completed_jobs),
            "cancelled_jobs": len(cancelled_jobs),
            "average_waiting_time": average_waiting_time,
            "average_execution_time": average_execution_time,
            "utilization": utilization,
            "allocation_efficiency": allocation_efficiency,
            "fragmentation": fragmentation,
            "deadlock_detected": self.detect_deadlock(),
            "scheduling_policy": self.scheduling_policy,
            "gpu_count": len(
                self.gpu_selector.get_gpu_ids()
            ),
            "gpu_utilization": (
                self.gpu_selector.get_gpu_utilization()
            ),
            "gpu_availability": (
                self.gpu_selector.get_gpu_availability()
            ),
            "preemption_count": self.preemption_count,
            "preempted_job_ids": list(self.preempted_job_ids),
            "preemptive_scheduling": (
                self.scheduling_policy == "PREEMPTIVE"
            )
        }

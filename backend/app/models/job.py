from datetime import datetime, timezone


class Job:

    def __init__(
        self,
        job_id,
        name,
        compute_required,
        memory_required,
        accelerator_a_required=0,
        accelerator_b_required=0,
        priority=1
    ):
        self.job_id = job_id
        self.name = name
        self.compute_required = compute_required
        self.memory_required = memory_required
        self.accelerator_a_required = accelerator_a_required
        self.accelerator_b_required = accelerator_b_required
        self.priority = priority

        self.status = "QUEUED"

        # GPU placement
        self.gpu_id = None

        self.queue_position = 0
        self.allocation_score = 0.0

        self.submitted_at = self._now()
        self.started_at = None
        self.waiting_started_at = None
        self.completed_at = None
        self.cancelled_at = None

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    def get_resource_request(self):
        return {
            "compute_units": self.compute_required,
            "memory": self.memory_required,
            "accelerator_a": self.accelerator_a_required,
            "accelerator_b": self.accelerator_b_required
        }

    def start(self):
        self.status = "RUNNING"
        if self.started_at is None:
            self.started_at = self._now()

        # Do not remove waiting_started_at here.
        # It is needed to calculate total waiting time correctly.

    def wait(self):
        self.status = "WAITING"
        if self.waiting_started_at is None:
            self.waiting_started_at = self._now()

    def complete(self):
        self.status = "COMPLETED"
        self.completed_at = self._now()

    def cancel(self):
        self.status = "CANCELLED"
        self.cancelled_at = self._now()

    def is_queued(self):
        return self.status == "QUEUED"

    def is_running(self):
        return self.status == "RUNNING"

    def is_waiting(self):
        return self.status == "WAITING"

    def is_completed(self):
        return self.status == "COMPLETED"

    def is_cancelled(self):
        return self.status == "CANCELLED"

    @staticmethod
    def _seconds_between(start, end):
        if start is None or end is None:
            return 0.0

        return round(
            max(
                0.0,
                (end - start).total_seconds()
            ),
            2
        )

    def get_queue_wait_seconds(self):
        if self.submitted_at is None:
            return 0.0

        if self.started_at is not None:
            return self._seconds_between(
                self.submitted_at,
                self.started_at
            )

        return self._seconds_between(
            self.submitted_at,
            self._now()
        )

    def get_waiting_duration_seconds(self):
        if self.waiting_started_at is None:
            return 0.0

        if self.started_at is not None:
            return self._seconds_between(
                self.waiting_started_at,
                self.started_at
            )

        if self.is_completed() or self.is_cancelled():
            end_time = (
                self.completed_at
                if self.completed_at is not None
                else self.cancelled_at
            )

            return self._seconds_between(
                self.waiting_started_at,
                end_time
            )

        return self._seconds_between(
            self.waiting_started_at,
            self._now()
        )

    def get_execution_duration_seconds(self):
        if self.started_at is None:
            return 0.0

        if self.completed_at is not None:
            return self._seconds_between(
                self.started_at,
                self.completed_at
            )

        if self.cancelled_at is not None:
            return self._seconds_between(
                self.started_at,
                self.cancelled_at
            )

        return self._seconds_between(
            self.started_at,
            self._now()
        )

    def get_total_duration_seconds(self):
        if self.submitted_at is None:
            return 0.0

        if self.completed_at is not None:
            end_time = self.completed_at

        elif self.cancelled_at is not None:
            end_time = self.cancelled_at

        else:
            end_time = self._now()

        return self._seconds_between(
            self.submitted_at,
            end_time
        )

    @staticmethod
    def _timestamp_to_string(timestamp):
        if timestamp is None:
            return None

        return timestamp.isoformat()

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "name": self.name,

            "compute_required": self.compute_required,
            "memory_required": self.memory_required,
            "accelerator_a_required": self.accelerator_a_required,
            "accelerator_b_required": self.accelerator_b_required,

            "priority": self.priority,

            # GPU placement
            "gpu_id": self.gpu_id,

            "queue_position": self.queue_position,

            "allocation_score": round(
                float(self.allocation_score),
                2
            ),

            "status": self.status,

            "submitted_at": self._timestamp_to_string(
                self.submitted_at
            ),

            "started_at": self._timestamp_to_string(
                self.started_at
            ),

            "waiting_started_at": self._timestamp_to_string(
                self.waiting_started_at
            ),

            "completed_at": self._timestamp_to_string(
                self.completed_at
            ),

            "cancelled_at": self._timestamp_to_string(
                self.cancelled_at
            ),

            "queue_wait_seconds":
                self.get_queue_wait_seconds(),

            "waiting_duration_seconds":
                self.get_waiting_duration_seconds(),

            "execution_duration_seconds":
                self.get_execution_duration_seconds(),

            "total_duration_seconds":
                self.get_total_duration_seconds()
        }

    def __repr__(self):
        return (
            f"Job("
            f"id={self.job_id}, "
            f"name='{self.name}', "
            f"status='{self.status}', "
            f"priority={self.priority}, "
            f"gpu='{self.gpu_id}'"
            f")"
        )
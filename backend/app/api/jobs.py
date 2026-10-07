from fastapi import APIRouter, HTTPException

from app.models.job import Job
from app.services.simulation import Simulation
from app.services.workload_generator import WorkloadGenerator
from app.services.stress_testing import StressTester


router = APIRouter(
    prefix="/jobs",
    tags=["Jobs"]
)


simulation = Simulation()
workload_generator = WorkloadGenerator()
stress_tester = StressTester()


def serialize_job(job):
    """Serialize a job and include the scheduler's allocation explanation."""
    data = job.to_dict()
    data["allocation_explanation"] = simulation.get_job_allocation_explanation(job)
    return data


@router.post("/")
def create_job(
    name: str,
    compute_required: int,
    memory_required: int,
    accelerator_a_required: int = 0,
    accelerator_b_required: int = 0,
    priority: int = 1
):
    if not name or not name.strip():
        raise HTTPException(
            status_code=400,
            detail="Job name cannot be empty."
        )

    if compute_required <= 0:
        raise HTTPException(
            status_code=400,
            detail="Compute requirement must be greater than 0."
        )

    if memory_required <= 0:
        raise HTTPException(
            status_code=400,
            detail="Memory requirement must be greater than 0."
        )

    if accelerator_a_required < 0:
        raise HTTPException(
            status_code=400,
            detail="Accelerator A requirement cannot be negative."
        )

    if accelerator_b_required < 0:
        raise HTTPException(
            status_code=400,
            detail="Accelerator B requirement cannot be negative."
        )

    if priority < 1:
        raise HTTPException(
            status_code=400,
            detail="Priority must be at least 1."
        )

    job_id = len(simulation.jobs) + 1

    job = Job(
        job_id=job_id,
        name=name.strip(),
        compute_required=compute_required,
        memory_required=memory_required,
        accelerator_a_required=accelerator_a_required,
        accelerator_b_required=accelerator_b_required,
        priority=priority
    )

    simulation.add_job(job)
    simulation.try_allocate_job(job)

    return {
        "message": "Job created successfully.",
        "job": serialize_job(job)
    }


@router.get("/")
def get_jobs():
    return {
        "jobs": [
            serialize_job(job)
            for job in simulation.get_jobs()
        ]
    }


@router.get("/statistics")
def get_statistics():
    return simulation.get_statistics()


@router.get("/historical-analytics")
def get_historical_analytics():
    return simulation.get_historical_analytics()


# ==========================================
# WORKLOAD GENERATOR
# ==========================================

@router.post("/workload/generate")
def generate_workload(
    count: int = 10,
    pattern: str = "MIXED",
    seed: int | None = None
):
    if count < 1 or count > 1000:
        raise HTTPException(
            status_code=400,
            detail="Workload count must be between 1 and 1000."
        )

    normalized_pattern = str(pattern).strip().upper()

    if normalized_pattern not in WorkloadGenerator.SUPPORTED_PATTERNS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid workload pattern.",
                "available_patterns": WorkloadGenerator.SUPPORTED_PATTERNS
            }
        )

    generator = WorkloadGenerator(seed=seed)
    specs = generator.generate(
        count=count,
        pattern=normalized_pattern,
        name_prefix="Generated Job"
    )

    created_jobs = []

    for spec in specs:
        job_id = simulation.job_sequence + 1

        job = Job(
            job_id=job_id,
            name=spec["name"],
            compute_required=spec["compute_required"],
            memory_required=spec["memory_required"],
            accelerator_a_required=spec["accelerator_a_required"],
            accelerator_b_required=spec["accelerator_b_required"],
            priority=spec["priority"]
        )

        simulation.add_job(job)
        simulation.try_allocate_job(job)
        created_jobs.append(job)

    result = {
        "message": "Workload generated successfully.",
        "pattern": normalized_pattern,
        "count": len(created_jobs),
        "jobs": [serialize_job(job) for job in created_jobs]
    }

    simulation.record_workload_batch({
        "pattern": normalized_pattern,
        "count": len(created_jobs),
        "seed": seed
    })

    return result


@router.post("/stress-test")
def run_stress_test(
    count: int = 100,
    pattern: str = "MIXED",
    seed: int | None = None
):
    if count not in StressTester.SUPPORTED_SIZES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid stress-test size.",
                "supported_sizes": StressTester.SUPPORTED_SIZES
            }
        )

    normalized_pattern = str(pattern).strip().upper()

    if normalized_pattern not in WorkloadGenerator.SUPPORTED_PATTERNS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid workload pattern.",
                "available_patterns": WorkloadGenerator.SUPPORTED_PATTERNS
            }
        )

    generator = WorkloadGenerator(seed=seed)
    specs = generator.generate(
        count=count,
        pattern=normalized_pattern,
        name_prefix="Stress Test Job"
    )

    result = stress_tester.run(
        simulation=simulation,
        workload_specs=specs,
        pattern=normalized_pattern
    )

    result["seed"] = seed
    result["supported_sizes"] = StressTester.SUPPORTED_SIZES

    simulation.record_stress_test(result)

    return {
        "message": "Stress test completed successfully.",
        "result": result
    }


@router.get("/workload/statistics")
def get_workload_statistics():
    return simulation.get_workload_statistics()


@router.get("/policy")
def get_scheduling_policy():
    return simulation.get_scheduling_policy()


@router.post("/policy")
def set_scheduling_policy(policy: str):
    success = simulation.set_scheduling_policy(policy)

    if not success:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid scheduling policy.",
                "available_policies":
                    simulation.SUPPORTED_POLICIES
            }
        )

    return {
        "message":
            "Scheduling policy updated successfully.",
        "policy":
            simulation.scheduling_policy,
        "available_policies":
            simulation.SUPPORTED_POLICIES
    }


@router.post("/{job_id}/complete")
def complete_job(job_id: int):
    job = next(
        (
            job for job in simulation.get_jobs()
            if job.job_id == job_id
        ),
        None
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    if not job.is_running():
        raise HTTPException(
            status_code=400,
            detail="Only a RUNNING job can be completed."
        )

    simulation.complete_job(job)

    return {
        "message": "Job completed successfully.",
        "job": job.to_dict()
    }


@router.post("/{job_id}/cancel")
def cancel_job(job_id: int):
    job = next(
        (
            job for job in simulation.get_jobs()
            if job.job_id == job_id
        ),
        None
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    if job.is_completed():
        raise HTTPException(
            status_code=400,
            detail="Completed jobs cannot be cancelled."
        )

    if job.is_cancelled():
        raise HTTPException(
            status_code=400,
            detail="Job is already cancelled."
        )

    success = simulation.cancel_job(job)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Unable to cancel this job."
        )

    return {
        "message": "Job cancelled successfully.",
        "job": job.to_dict()
    }

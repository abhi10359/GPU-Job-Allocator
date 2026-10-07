from fastapi import APIRouter

from app.api.jobs import simulation

router = APIRouter(
    prefix="/resources",
    tags=["Resources"]
)


@router.get("/")
def get_resources():
    return simulation.get_resource_analysis()


@router.get("/deadlock")
def get_deadlock_status():

    deadlock_detected = simulation.detect_deadlock()

    return {
        "deadlock_detected": deadlock_detected,
        "status": (
            "DEADLOCK DETECTED"
            if deadlock_detected
            else "SAFE"
        )
    }
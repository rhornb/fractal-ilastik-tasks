"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ParallelTask

AUTHORS = "Lorenzo Cerrone"
DOCS_LINK = "https://github.com/fractal-analytics-platform/fractal-ilastik-tasks"

TASK_LIST = [
    ParallelTask(
        name="Ilastik Pixel Classification Segmentation",
        executable="ilastik_pixel_classification_segmentation.py",
        meta={"cpus_per_task": 8, "mem": 8000},
        category="Segmentation",
        tags=[
            "Pixel Classifier",
        ],
        docs_info="file:docs_info/ilastik_pixel_classifier.md"
    ),
]

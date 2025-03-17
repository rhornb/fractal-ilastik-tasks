"""Generate JSON schemas for task arguments."""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "ilastik_tasks"
    AUTHORS = "Lorenzo Cerrone"
    create_manifest(package=PACKAGE, authors=AUTHORS)

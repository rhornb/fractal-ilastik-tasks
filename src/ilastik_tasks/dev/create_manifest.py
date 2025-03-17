"""Generate JSON schemas for task arguments."""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "ilastik_tasks"
    AUTHORS = "Lorenzo Cerrone"
    docs_link = "https://github.com/fractal-analytics-platform/fractal-ilastik-tasks"
    create_manifest(package=PACKAGE, authors=AUTHORS, docs_link=docs_link)

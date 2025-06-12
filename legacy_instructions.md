# Installation and Deployment

* Install the `mamba` package manager

* Download the installation script from this repository

```bash
curl -O https://raw.githubusercontent.com/fractal-analytics-platform/fractal-ilastik-tasks/main/create_env_script.sh
```

* The scrip might require some small modifications.

```bash
VERSION="v0.2.0" # Version of the package to install (by default the latest version)
COMMMAND="mamba" # Command to use to create the environment (mamba or conda) 
# Location of the environment
# If ENVPREFIX is not NULL, the environment will be created with the prefix $ENVPREFIX/$ENVNAME 
# If ENVPREFIX is NULL, the environment will be created in the default location
ENVPREFIX="NULL" 
```

* Install the package using the installation script
  
```bash
bash create_env_script.sh
```

The installation script will create a conda environment with the name `fractal-ilastik-tasks` and install the package in the environment. It will also download the correct `__FRACTAL_MANIFEST__.json` file.

* In the fractal web interface add the task to the workflow as a "local env" task.

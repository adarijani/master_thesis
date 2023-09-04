# How to run your code on the cluster?

A brief tutorial for using dell machine in room 124

## 0. Preparation
* Request VPN access (email: helpdesk@cats.rwth-aachen.de) with TIM id and expiration date in case of HiWis and master students.

* Download Cisco AnyConnect

## 1. Transfer your files and run your code on a cluster

*  Connect to VPN with Cisco Anyconnect (vpn.ims.rwth-aachen.de) and login with your TIM ID (i.e. ab123456) either by using the GUI or by running

    ```$ /opt/cisco/anyconnect/bin/vpn connect vpn.ims.rwth-aachen.de <<"EOF"```

    in a terminal.

### 1.1. For file transfer with Finder

* Please make sure to put your data on /mnt/Data/<-lastname-> (i.e. smith) 

* Finder -> Go -> Connect to server (smb://msip-dell.ims.rwth-aachen.de) and login with your last name (i.e. smith)

* Connect to the share point "data"

### 1.2. Run Your code in terminal

    $ ssh msip-dell.ims.rwth-aachen.de

### 1.3. Creating an SSH tunnel to access the machine from RWTH eduroam or normal RWTH VPN (vpn.rwth-aachen.de).

* Create the tunnel through the MacPro (student200).

    ```$ ssh student200.aices.rwth-aachen.de -L 22222:msip-dell.ims.rwth-aachen.de:22 -L 44445:msip-dell.ims.rwth-aachen.de:445```

    This can be configured in the SSH config as well, see [How to config SSH](How_to_config_SSH.md)

    In case `student200` is down, any other machine in the AICES network can be used to tunnel through, e.g. `nre.aices.rwth-aachen.de`. This is another MacPro that is permanently running.

* Use ssh through the tunnel

    ```$ ssh localhost -p 22222```

* Moreover `localhost:44445` can be used for samba access.

## 2. Tips:

### Conda commands:

Install conda

    $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh

Update conda

    $ conda update --all

Create a conda environment

    $ conda create -n myenv

Activate a conda environment

    $ conda activate myenv

Remove an entire conda environment

    $ conda remove --name fenicsproject --all

List all conda environments

    $ conda info --envs

[more info on Conda](https://git-ce.rwth-aachen.de/msip/group-repo/-/blob/master/wiki/Conda_tips.md)
### Git commands:
 
Configure the author name and email address to be used with your commits

    $ git config --global user.name "Sam Smith"
    $ git config --global user.email sam@example.com

            
Create a new local repository

    $ git init

Create a working copy of a local repository

    $ git clone /path/to/repository

Stage changes

    $ git add .

Commit changes

    $ git commit -m "Commit message"

Push to remote repository

    $ git push

Fetch and merge changes on the remote server to your working directory

    $ git pull

### scp

Copy a file from your local machine to the dell maching using scp

    $ scp output-full.mp4 berkels@msip-dell.ims.rwth-aachen.de:/mnt/Data/berkels/Downloads/


### podman (the Docker alternative on Linux)

**You need to setup a Dockerfile first**

Build a container with name

    $ podman build -t <name> <path to Dockerfile>

Run a container

    $ podman run -it <name>

List containers

    $ podman container ls -a

List images 

    $ podman image ls -a

Fancy cleanup to remove ALL images and containers

    $ podman stop $(podman ps -a -q);podman rm $(podman ps -a -q);podman rmi $(podman image ls -a -q)

### Helpful config settings in podman

Settings in `~/.config/containers/libpod.conf`

    cgroup_manager = "cgroupfs"
    volume_path = "/mnt/Data/<lastname>/containers/storage/volumes"
    static_dir = "/mnt/Data/<lastname>/containers/storage/libpod"

Settings in `~/.config/containers/storage.conf`

    driver = "vfs"
    runroot = "/mnt/Data/<lastname>/run/containers/storage"
    graphroot = "/mnt/Data/<lastname>/run/containers/storage"
    rootless_storage_path = "/mnt/Data/<lastname>/containers/storage"

## Run your Jupiter notebook on a server

Connect to MSIP Dell computer with ssh

    $ ssh msip-dell.ims.rwth-aachen.de

Launch Jupyter Notebook from remote server using port 8080

    $ jupyter notebook --no-browser --port=8080

Remember the given **token**

Open new terminal window on your local machine, run

    $ ssh -L 8080:localhost:8080 lastname@msip-dell

and enter your password

Open a browser from your local machine and navigate to

    http://localhost:8080/

Enter the **token**






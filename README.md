# synthetic-biometric-data-generation

### Clone
* `git clone --recurse-submodules https://github.com/blastak/synthetic-biometric-data-generation.git`

### Environment settings
#### requirements
* `torch > 2.0.0` for `torch.set_default_device()`
#### case1: conda
* `conda create --name synthetic-biometric-data-generation python=3.8.16`
* `conda activate synthetic-biometric-data-generation`
* `pip install torch --index-url https://download.pytorch.org/whl/cu118`
* `pip install torchvision`
* `pip install opencv-python==4.7.0.72`
* `pip install pythonnet`
* `pip install circle_fit`
#### case2: docker
* docker setup
  * `docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`
  * `cd ~/<MY_DIRECTORY>`
  * `docker run -it -v $(pwd):/workspace --net=host --privileged --gpus all --ipc=host --name=<CONTAINER_NAME>/pytorch:2.0.1-cuda11.7-cudnn8-devel`
* inside container
  * `pip install opencv-python-headless` this is for CLI-based server, **not** GUI-based
  * `pip install pythonnet`
  * `pip install circle_fit`

 
* ::::::::::::: (OPTIONAL INFORMATION) Docker container handling cheatsheet ::::::::::::
  * 블로그 참고
    * https://blastak.github.io/도커-쉽게-사용하는-방법-(나의-경우).html
  * (inside) To exit without stopping the container
    * `ctrl+p+q` or just press button [X] in your terminal
  * (outside) To check current status
    * `docker ps -a`
  * (outside) To re-enter **running** container
    * `docker attach <CONTAINER_NAME>`
  * To exit with STOP
    * (inside) `exit`
    * (outside) `docker stop <CONTAINER_NAME>`
  * To start **exited** container
    * (outside) `docker start -ai <CONTAINER_NAME>`

### Training
T.B.D

### Generating synthetic biometric samples
T.B.D

### Matching
T.B.D

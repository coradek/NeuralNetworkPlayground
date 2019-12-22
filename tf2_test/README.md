# README for tf2_test project

### _SYNC WITH COMPUTE_
- All changes to src/ must be made locally
- All changes to notebooks/ must be made on $COMPUTE
- Other changes to these directories will be erased during sync
`./sync_w_columbus.sh`

## __SETUP:__

### If Anaconda is not already installed
```
# copy Anaconda setup script to REMOTE
scp /Users/i862304/workspace/_remote_setup/Anaconda3-2019.03-Linux-x86_64.sh $REMOTE:~/

# (on REMOTE)
cd ~/
chmod 700 Anaconda3-2019.03-Linux-x86_64.sh
./Anaconda3-2019.03-Linux-x86_64.sh
# (install to /opt/anaconda3)

# (downgrade python for tf compatibility)
conda install python=3.6.5

echo 'export PATH=/opt/anaconda3/bin/python:$PATH' >> ~/.bash_profile
source ~/.bash_profile
```

### Build place for tf2 work

__Install Tensorflow 2.0__

mkdir for tf2 work
cd into dir
`python -m venv tf2env`

activate tf2 env
`source ./tf2env/bin/activate`

(as of writing, TF website recommends
    `pip install tensorflow-gpu==2.0.0-rc0`
    which is not available in artifactory)

`pip install tensorflow 2.0 (currently nightly-build beta)
sudo pip install tf-nightly-gpu-2.0-preview --index-url https://artifactory.concurtech.net/artifactory/api/pypi/pypi-sandbox/simple`

(what are the differences between either of the above and
    tensorflow-gpu==2.0.0b1)

verify install
`python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`


`sudo pip install jupyter --index-url https://artifactory.concurtech.net/artifactory/api/pypi/pypi-sandbox/simple`



## __NOTES:__

installing packages in AWS-Prod w/o access to pypi

on local
`pip download --platform linux_x86_64 --only-binary=:all: --python-version 3 numpy`

copy .whl to REMOTE

on REMOTE:
pip install path/to/whl

# syntax=docker/dockerfile:1

# Sets up the working base for all subcontainers
FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3 AS mime-base
ENV PROGHOME=/mime
ENV DEFAULTSCRIPT=init.sh



# Bring current
RUN apt-get update -qq
RUN apt-get install -y -qq && apt-get dist-upgrade -y -qq

# Add VPI Support
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc
ENV NVREPOLIST=/etc/apt/sources.list.d/xavier.list
RUN echo "# NVidia Jetson Xavier packages for hardware acceleration." > ${NVREPOLIST}
RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.6 main" >> ${NVREPOLIST}
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 r32.6 main" >> ${NVREPOLIST}


# Add ROS2 Foxy support
### BEGIN SELECTIVE COPY ###
# Basically directly copied from: https://github.com/dusty-nv/jetson-containers/blob/master/Dockerfile.ros.foxy
ENV ROS_PKG=ros_base
ENV ROS_DISTRO=foxy
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"] 
WORKDIR /tmp
# change the locale from POSIX to UTF-8
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8
# add the ROS deb repo to the apt sources list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
# install development packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		build-essential \
		cmake \
		git \
		libbullet-dev \
		libpython3-dev \
		python3-colcon-common-extensions \
		python3-flake8 \
		python3-pip \
		python3-numpy \
		python3-pytest-cov \
		python3-rosdep \
		python3-setuptools \
		python3-vcstool \
		python3-rosinstall-generator \
		libasio-dev \
		libtinyxml2-dev \
		libcunit1-dev \
		libgazebo9-dev \
		gazebo9 \
		gazebo9-common \
		gazebo9-plugin-base \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
# install some pip packages needed for testing
RUN python3 -m pip install -U \
    argcomplete \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-docstrings \
    flake8-import-order \
    flake8-quotes \
    pytest-repeat \
    pytest-rerunfailures \
    pytest
# compile yaml-cpp-0.6, which some ROS packages may use (but is not in the 18.04 apt repo)
RUN git clone --branch yaml-cpp-0.6.0 https://github.com/jbeder/yaml-cpp yaml-cpp-0.6 && \
    cd yaml-cpp-0.6 && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && \
    cp libyaml-cpp.so.0.6.0 /usr/lib/aarch64-linux-gnu/ && \
    ln -s /usr/lib/aarch64-linux-gnu/libyaml-cpp.so.0.6.0 /usr/lib/aarch64-linux-gnu/libyaml-cpp.so.0.6
# download/build ROS from source
RUN mkdir -p ${ROS_ROOT}/src && \
    cd ${ROS_ROOT} && \
    # https://answers.ros.org/question/325245/minimal-ros2-installation/?answer=325249#post-id-325249
    rosinstall_generator --deps --rosdistro ${ROS_DISTRO} ${ROS_PKG} \
		launch_xml \
		launch_yaml \
		launch_testing \
		launch_testing_ament_cmake \
		demo_nodes_cpp \
		demo_nodes_py \
		example_interfaces \
		camera_calibration_parsers \
		camera_info_manager \
		cv_bridge \
		v4l2_camera \
		vision_opencv \
		vision_msgs \
		image_geometry \
		image_pipeline \
		image_transport \
		compressed_image_transport \
		compressed_depth_image_transport \
		> ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
    cat ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
    vcs import src < ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
    # patch libyaml - https://github.com/dusty-nv/jetson-containers/issues/41#issuecomment-774767272
    rm ${ROS_ROOT}/src/libyaml_vendor/CMakeLists.txt && \
    wget --no-check-certificate https://raw.githubusercontent.com/ros2/libyaml_vendor/master/CMakeLists.txt -P ${ROS_ROOT}/src/libyaml_vendor/ && \
    # install dependencies using rosdep
    apt-get update && \
    cd ${ROS_ROOT} && \
    rosdep init && \
    rosdep update && \
    rosdep install -y \
    	  --ignore-src \
       --from-paths src \
	  --rosdistro ${ROS_DISTRO} \
	  --skip-keys "libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv" && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    # build it!
    colcon build --merge-install && \
    # remove build files
    rm -rf ${ROS_ROOT}/src && \
    rm -rf ${ROS_ROOT}/logs && \
    rm -rf ${ROS_ROOT}/build && \
    rm ${ROS_ROOT}/*.rosinstall
# Set the default DDS middleware to cyclonedds
# https://github.com/ros2/rclcpp/issues/1335
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Finish up installation
COPY ./scripts/ros_entrypoint.sh /ros_entrypoint.sh
RUN sed -i 's/ros_env_setup="\/opt\/ros\/$ROS_DISTRO\/setup.bash"/ros_env_setup="${ROS_ROOT}\/install\/setup.bash"/g' \
    /ros_entrypoint.sh && \
    cat /ros_entrypoint.sh
RUN echo 'source ${ROS_ROOT}/install/setup.bash' >> /root/.bashrc
### END SELECTIVE COPY ###


# Finish up and turn into a command
WORKDIR ${PROGHOME}
COPY config/. ${PROGHOME}/config/
ENTRYPOINT chmod +x ${PROGHOME}/${DEFAULTSCRIPT} && \
    bash --init-file ${PROGHOME}/${DEFAULTSCRIPT}



### LATER BUILD STAGES AFTER THIS POINT ###



## Build dependencies defined first ##
FROM mime-base as mime-base-installing
RUN apt-get update -qq
#ENV VPI_BASE_DEPENDENCIES="libnvvpi1 vpi1-dev"
ENV CAPTURE_DEPENDENCIES="python3-vpi1 alsa-base alsa-utils"
ENV FACIAL_DEPENDENCIES="golang-1.13 golang-1.13-doc golang-1.13-go golang-1.13-src"
ENV TERMINAL_DEPENDENCIES="magic-wormhole git"


# Enables terminal access
FROM mime-base-installing as mime-terminal
RUN apt-get install -y ${CAPTURE_DEPENDENCIES}
RUN apt-get install -y ${FACIAL_DEPENDENCIES}
RUN apt-get install -y ${TERMINAL_DEPENDENCIES}
COPY terminal/. ${PROGHOME}/
COPY . ${PROGHOME}/source/


# Captures data from physical and virtual sensors
FROM mime-base-installing AS mime-capture
RUN apt-get install -y -qq ${CAPTURE_DEPENDENCIES}
COPY perception/. ${PROGHOME}/


# Displays facial features
FROM mime-base-installing as mime-face
RUN apt-get install -y -qq ${FACIAL_DEPENDENCIES}
COPY face/. ${PROGHOME}/


# Run main ROS processes and JAMES functionality
FROM mime-base AS mime-brain
COPY brain/. ${PROGHOME}


# Wrangle MIMe attachments
FROM mime-base AS mime-limbs
COPY limbs/. ${PROGHOME}/

FROM r-base:3.4.4
# https://github.com/rocker-org/rocker/blob/151274f9052f874034489769b94eba834309dc5d/r-base/Dockerfile


##https://mxnet.incubator.apache.org/install/index.html
RUN gcc --version && make --version
RUN apt-get update \
  && apt-get -y upgrade \
  && apt-get install -y build-essential git libopenblas-dev liblapack-dev libopencv-dev \
    libssl-dev libcurl4-openssl-dev libxml2-dev

##NOTE: In this step we install OpenBLAS. You can choose to install ATLAS or MKL

RUN cd ~ \
  && git clone --branch 1.2.0.rc1 --recurse https://github.com/apache/incubator-mxnet \
  && cd ~/incubator-mxnet \
  && make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas

RUN  Rscript -e 'install.packages(c("devtools"), repos = "http://cran.us.r-project.org")' #,"curl","rgexf","DiagrammeR"
RUN cd ~/incubator-mxnet && make rpkg

COPY . .

#RUN sed 's/array.layout=array.layout/array.layout=array.layout,/g' < test.R > test_broken.R

CMD Rscript test.R && Rscript test_broken.R

## (c) 2007 The Board of Trustees of the University of Illinois.
#
#LANGUAGE=cuda
#
#SRCDIR_OBJS=file.o main.o
#APP_LDFLAGS=-lm -lstdc++
#APP_CUDACFLAGS=--use_fast_math

################################################################################
#
# Build script for project
#
################################################################################
# Add source files here
EXECUTABLE	:= mri-q
# Cuda source files (compiled with cudacc)
CUFILES		:= main.cu
# Cuda dep
CU_DEPS 	:= computeQ.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= file.cc parboil.c

################################################################################
# Rules and targets

include /gpgpu/common/common.mk

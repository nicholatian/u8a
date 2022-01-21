##############################################################################
##                                   u8a.                                   ##
##                                                                          ##
##                   Copyright (C) 2021 Alexander Nicholi                   ##
##                           All rights reserved.                           ##
##############################################################################

include etc/base.mk

# name of project. used in output binary naming
PROJECT := u8a

# put a ‘1’ for the desired target types to compile
EXEFILE :=
SOFILE  :=
AFILE   := 1

# space-separated path list for #includes
# <system> includes
INCLUDES := include
# "local" includes
INCLUDEL := src

# space-separated library name list
LIBS    :=
LIBDIRS :=

# ‘3P’ are in-tree 3rd-party dependencies
# 3PLIBDIR is the base directory
# 3PLIBS is the folder names in the base directory for each library
3PLIBDIR :=
3PLIBS   :=

# frameworks (macOS target only)
FWORKS :=

# sources
CFILES    := \
	src/main.c
CPPFILES  :=
PUBHFILES :=
PRVHFILES :=

# test suite sources
TES_CFILES    :=
TES_CPPFILES  :=
TES_PUBHFILES :=
TES_PRVHFILES :=

CFLAGS := $(shell python3-config --include)

# this defines all our usual targets
include etc/targets.mk

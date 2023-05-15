# Just change SHCI_DIR to the location of the SHCI directory
SHCI_DIR := ../..

LIB_DIR := $(SHCI_DIR)/lib
PARENT_DIR := $(SHCI_DIR)/..
SRC := unpack_wf.cc
UTIL := $(SHCI_DIR)/src/util.cc
HDET := $(SHCI_DIR)/src/det/half_det.cc

all : unpack_wf.cc
	g++ -I$(LIB_DIR) -I$(PARENT_DIR) $(UTIL) $(HDET) $(SRC) -fopenmp -o unpack_wf

clean :
	$(RM) unpack_wf

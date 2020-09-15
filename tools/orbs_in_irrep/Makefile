FC=gfortran
FFLAGS=-finit-local-zero -O3 -ffixed-line-length-132 -Wall -g -fbounds-check -fbacktrace
F90FLAGS = -finit-local-zero -O3 -ffree-line-length-none -x f95-cpp-input -Wall -g -fbounds-check -fbacktrace

.SUFFIXES:
.SUFFIXES: .f90 .f95 .o .f .c

.f90.o:
	$(FC) $(F90FLAGS) -o $@ -c $<

OBJS =	c.o

orbs_in_irrep: $(OBJS)
	$(FC) $(FFLAGS) $(LDFLAGS) $(OBJS) $(LIBS) -o $@

cell_clustering
===============

Overview
--------

The cell clustering problem is an example for agent-based
self-organization, comprising many cell behaviours occurring naturally
during neural development. During development of the cerebral cortex
(i.e.  the part of the brain that is largely responsible for
sensation, action, cognition and memory), a few initial progenitor
cells proliferate through division and cell growth, giving rise to an
exponentially increasing population of different cell types. These
cells, by sensing extracellular substances in their environment,
migrate towards their final locations.  Hence, by implementing
optimized code for simulating cell proliferation, migration and
diffusion of substances, it will be possible to efficiently conduct
large-scale simulations of cortical development. Such simulations are
crucial to gain insights into neurodevelopmental disorders, and the
workings of the brain.

Contents
--------

### Source files ###

- cell_clustering.cpp

   The main source file, containing the main() function and all code
   to be measured. Only this source file need be modified in the
   optimization effort.

- util.hpp

   A header file of utility functions.

- util.cpp

   A file containing the implementation of utility functions used in
   the code.

### Build files ###

- Makefile

   The Makefile for the project. Contest entries are welcome to modify
   the options to the compiler therein as part of their optimization
   efforts

- BUILD-HOST-GEN

   A script that records information about the configuration of the
   machine it is run on.

### Input files ###


- small.cdc

   A small-sized problem suitable for quick performance
   and correctness tests

- huge.cdc

   A very large problem for final performance runs. The serial version
   of the code included here will take a very, very long time to run
   this code as it is, and this should be only run with an optimized
   version of the code. You won't be able to run the code on this input 
   on the cluster as it will timeout.  



### Miscellaneous ###

- LICENSE

   Describes the license covering this code

- README.md

   This file.

Building
--------

The build system for this code is very simple; assuming a modern
"make" command is present on your system, simply run

    $ make cell_clustering

To build the code with the default compiler (the value of CXX in your
environment). To build with a different compiler, set up the build
environment as per your compiler vendor's instructions and run:

    $ make CXX=<C++ compiler>

Where <C++ compiler> is the command name of your C++ compiler.

    $ make clean

This will delete the binary cell_clustering.

This version of the code is known to work with gcc-g++ version 4.8.3
and with Intel(R) C++ Compiler 2016 (build 20150815) with glibc 2.17.

To compile this code for Intel(R) Xeon Phi(TM), it will be necessary
to change the compiler switches in the Makefile, replacing

    -xHost

with

    -mmic

Running
-------

- On you machine
  ---------------
The cell_clustering binary accepts multiple command-line options that are
documented with

    $ ./cell_clustering --help

At a minimum, it should be invoked with an input script akin to the
{small or huge}.cdc files distributed here.

It is possible to override input parameters in an input script with
the

    --<param>=<value>

syntax, which may be useful for temporarily running fewer timesteps,
for example.

cell_clustering writes output to both stdout and stderr as it runs.  The
information written to stdout may be increased or decreased by
providing (possibly multiples of) the -q or -v switches to the command
line.

The output to stderr is designed to capture the configuration of the
code, the input parameters used, the performance, and the correctness
of each run in a machine-readable format.

- On the cluster
 ----------------
Please refer to the instructions on how to run the code on the cluster.


### Correctness ###

Particular attention should be paid to the FINAL\_CRITERION value that
is printed to stderr at the completion of the code; it will be '1' if
the run produced acceptable results, and '0' if it did not. *Only
those entries that successfully produce FINAL\_CRITERION = 1 for the
huge.cdc input file will be considered for this contest*.

### Performance ###

Much of the output written to stderr is aimed a reporting the
performance of the code.  Most of it is merely intended to be
informative to the user, but the TOTAL\_COMPUTE\_TIME value, which
reports the wall time of the code from just before Phase 1 begins
until Phase 2 finishes will be used to determine each contest entries'
overall performance. Note that this time excludes the initialization,
finalization, and reporting times of the program.
To win the contest, the code needs to generate the minimum 
TOTAL\_COMPUTE\_TIME value using huge.cdc as input file.

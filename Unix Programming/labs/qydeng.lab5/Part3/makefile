all: my_npipe_reader my_npipe_writer

CC = gcc
CFLAGS = -Wall -g -c

my_npipe_reader: my_npipe_reader.o
my_npipe_writer: my_npipe_writer.o

my_npipe_reader.o: my_npipe_reader.c
my_npipe_writer.o: my_npipe_writer.c

clean:
	rm *.o my_npipe_reader my_npipe_writer

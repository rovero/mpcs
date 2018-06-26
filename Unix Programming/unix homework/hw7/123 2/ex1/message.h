#define MAXSPUDS 30000

#define LEN 256
struct message {
	long msg_size;
	char* content; /* even is white, odd is red, mod 5 is "rotten" */
};

char posix_potato [BUFSIZ];


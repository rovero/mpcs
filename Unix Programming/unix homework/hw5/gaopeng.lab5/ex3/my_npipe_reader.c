#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CHILD (pid_t) 0

int main(int argc, char * argv[] )
{
	static char message[BUFSIZ];
	char * myfifo = "/tmp/mypipe";
	int fifo;
	int n;

	/*fprintf(stderr, "USAGE:  %s >/dev/tty/?\n",argv[0]);*/

	/* create the FIFO, if it already exists, ignore the error */
	if( mkfifo(myfifo, 0666) < 0 )
	{
		fprintf(stderr,"it seems %s already exists, I'll use it\n",myfifo);
	}
    printf("Creating named pipe: %s\n",myfifo);

	/* now, let's open the FIFO */
	if( (fifo = open(myfifo, O_RDONLY)) < 0)
	{
		perror("oh oh:");
		exit(1);
	}
    printf("Waiting for input...");
    fflush( stdout );
    char *_exit = "exit";
	while( (n = read(fifo, message, sizeof(message))) > 0 )
	{
		fprintf(stderr,"Got it: '%s'\n",message);
        if(strcmp(_exit,message)==0)
            break;
		memset(message, '\0', BUFSIZ);
        printf("Waiting for input...");
        fflush( stdout );
	}
	putchar('\n');
    printf("Exiting\n");
	close(fifo);
	exit(0);
}


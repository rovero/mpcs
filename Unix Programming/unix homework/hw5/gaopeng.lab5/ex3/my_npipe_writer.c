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

	/*fprintf(stderr, "USAGE:  %s >/dev/tty/?\n",argv[0]);*/

	/* create the FIFO, if it already exists, ignore the error */


	/* now, let's open the FIFO */
	if( (fifo = open(myfifo, O_WRONLY)) < 0)
	{
		perror("oh oh:");
		exit(1);
	}
	else
		printf("FIFO opened successfully, talk to me baby...\n");

    
    printf("Opening named pipe: %s\n",myfifo);
    char *_exit = "exit\n";
    printf("Enter Input:");
	while ( fgets(message,BUFSIZ,stdin) != NULL )
	{
		write(fifo, message, strlen(message)-1);
        printf("Writing buffer to pipe...done\n");
        if(strcmp(_exit,message)==0)
            break;
		memset(message, '\0', BUFSIZ);
        printf("Enter Input:");
	}
	putchar('\n');
    printf("Exiting\n");
	close(fifo);
	exit(0);
}


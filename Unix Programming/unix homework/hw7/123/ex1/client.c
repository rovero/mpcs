#include <sys/types.h>
#include <sys/ipc.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "message.h"





int main(int argc, char * argv [] )
{
	struct message msg;
	int retval;
	int randval;
	int y;
	int sleeping = 0;
	int msgid,msgid2;
	int x;
	static int rotten, white, red;

	msgid = msgget((key_t) 73707564, 0666 | IPC_CREAT);
    msgid2 = msgget((key_t) 73707565, 0666 | IPC_CREAT);
	if( msgid == -1 )
	{
		fprintf(stderr, "damn!  msgget failed\n");
		exit(EXIT_FAILURE);
	}
    if( msgid2 == -1 )
    {
        fprintf(stderr, "damn!  msgget failed\n");
        exit(EXIT_FAILURE);
    }

	for( ; ; )
	{
        
        if(retval){
                fprintf(stderr, "damn!  msgget failed\n");
                exit(EXIT_FAILURE); /* failed to read any more */
        }
        
        printf("Insert message to send to server: ");
        
        size_t size;
        char *line = NULL;
        if (getline(&line, &size, stdin) == -1) {
            printf("No line\n");
        }
        
        msg.msg_size = (long int)strlen(line)-1;
        msg.content = malloc((strlen(line)-1)*sizeof(char*));
        memcpy(msg.content,line,strlen(line)-1);

        
        if( msgsnd(msgid, (void *) msg.content, strlen(line)-1, 0) == -1 )
        {
            fprintf(stderr, "msgsnd failed\n");
            exit(EXIT_FAILURE);
        }

        char *buffer = (char *)malloc(LEN);
        if( msgrcv(msgid2, (void *) buffer, LEN, 0, 0) == -1 )
        {
            fprintf(stderr, "msgsnd failed\n");
            exit(EXIT_FAILURE);
        }
        
        printf("Msg processed: %s\n",buffer);

	}
}


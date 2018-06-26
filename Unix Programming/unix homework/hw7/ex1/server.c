#include <sys/types.h>
#include <sys/ipc.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "message.h"



/* convert upper case to lower case or vise versa */
void conv(char *msg, int size)
{
    int i=0;
    for (i=0; i<size; ++i)
        if (islower(msg[i]))
            msg[i] =  toupper(msg[i]);
        else if (isupper(msg[i]))
            msg[i] =  tolower(msg[i]);
}



int main(int argc, char * argv [] )
{
	int randval;
    struct message msg;
	int msgid, msgid2;
	int x;
	
	time_t t;
	int maxspuds = MAXSPUDS;


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
    char *buffer = (char *)malloc(LEN);
    memset(buffer,0,LEN);
    
    while(1){
        msg.msg_size=1;
        if( msgrcv(msgid, (void *) buffer, LEN, 0,0) == -1 )
        {
            fprintf(stderr, "msgsnd failed\n");
            exit(EXIT_FAILURE);
        }
        //printf("%s len = %d\n",buffer,strlen(buffer));
        conv(buffer,strlen(buffer));
        //printf("after convertion, string is %s\n",buffer);
        
        
        

        if( msgsnd(msgid2, (void *) buffer, strlen(buffer), 0) == -1 )//IPC_NOWAIT
        {
            fprintf(stderr, "msgsnd failed\n");
            exit(EXIT_FAILURE);
        }
    }
    
}


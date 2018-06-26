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
        if( msgrcv(msgid, (void *) buffer, LEN, 0) == -1 )
        {
            fprintf(stderr, "msgsnd failed\n");
            exit(EXIT_FAILURE);
        }
        //printf("%s len = %d\n",buffer,strlen(buffer));
        conv(buffer,strlen(buffer));
        //printf("after convertion, string is %s\n",buffer);
        
        
        

        if( msgsnd(msgid2, (void *) buffer, strlen(buffer), 0) == -1 )
        {
            fprintf(stderr, "msgsnd failed\n");
            exit(EXIT_FAILURE);
        }
    }
    
    /*
	for( x = 0; x < maxspuds; x++ )
	{
		spud.msg_type = 1;
		randval = rand();
		if ( randval == 0 )
			randval++;
		if( (randval % 5) == 0 )
		{
			spud.type = ROTTEN;
			rotten++;
		}
		else if( (randval % 2) == 0 )
		{
			spud.type = WHITE;
			white++;
		}
		else
		{
			spud.type = RED;
			red++;
		}
		if( msgsnd(msgid, (void *) &spud, sizeof(struct potato), 0) == -1 )
		{
			fprintf(stderr, "msgsnd failed\n");
			exit(EXIT_FAILURE);
		}
		else
			printf("added message index %d: spud.type is %d\n",x,spud.type);
	}
	printf("I created %d potatoes, of which:\n%d were white\n%d were red\nand %d were rotten\n",white+red+rotten,white,red,rotten);
    
    */
}


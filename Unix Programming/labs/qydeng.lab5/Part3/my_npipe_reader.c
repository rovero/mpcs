//
//  my_npipe_reader.c
//  lab5
//
//  Created by Qianyu Deng on 2017/2/5.
//  Copyright © 2017年 Qianyu Deng. All rights reserved.
//

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
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
    printf("Creating named pipe: /tmp/mypipe\n");
    /* create the FIFO, if it already exists, ignore the error */
    if( mkfifo(myfifo, 0666) < 0 )
    {
        fprintf(stderr,"it seems %s already exists, I'll use it\n",myfifo);
    }
    
    /* now, let's open the FIFO */
    if( (fifo = open(myfifo, O_RDONLY)) < 0)
    {
        perror("oh oh:");
        exit(-1);
    }
    printf("Waiting for input...");
    while(1)
    {
        if((n=read(fifo, message, sizeof(message)))>0){
            printf("Got it: '%s'\n",message);
        }
        if(strcmp(message,"exit")==0){
            break;
        }
        memset(message, '\0', BUFSIZ);
        printf("Waiting for input...");
    }
    printf("Exiting\n");
    close(fifo);
    exit(0);
}

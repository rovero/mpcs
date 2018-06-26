//
//  my_npipe_writer.c
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
    int retval;
    
    /*fprintf(stderr, "USAGE:  %s >/dev/tty/?\n",argv[0]);*/
    
    /* create the FIFO, if it already exists, ignore the error */
    printf("Opening named pipe: /tmp/mypipe\n");
    /* now, let's open the FIFO */
    if( (fifo = open(myfifo, O_WRONLY)) < 0)
    {
        perror("oh oh:");
        exit(-1);
    }
    
    while (1)
    {
        printf("Enter Input: ");
        fgets(message,BUFSIZ,stdin);
        //message[strlen(message)]='\0';
        retval = write(fifo, message, strlen(message)-1);
        printf("Writing buffer to pipe...done\n");
        if(strcmp(message,"exit\n")==0){
            break;
        }
        memset(message, '\0', BUFSIZ);
    }
    printf("Exiting\n");
    close(fifo);
    exit(0);
}

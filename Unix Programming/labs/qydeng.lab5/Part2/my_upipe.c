//
//  my_upipe.c
//  lab5
//
//  Created by Qianyu Deng on 2017/2/5.
//  Copyright © 2017年 Qianyu Deng. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <string.h>
#define CHILD (pid_t) 0
void tokenizing(char* line,char** arg);

int main(int argc,char** argv){
	pid_t child;
	int apipe[2];
	int i=0;
	if(argc!=3){
		printf("Usage: ./my_upipe cmd1 cmd2\n");
		exit(1);
	}
	if(pipe(apipe)==-1){
		printf("fail to create a pipe\n");
		exit(-1);	
	}
	if((child=fork())==CHILD){
		//printf("apipe0: %d\n",apipe[0]);
		//printf("apipe1: %d\n",apipe[1]);
		close(apipe[0]);
		dup2(apipe[1],fileno(stdout));
		//printf("apipe1: %d\n",apipe[1]);
		close(apipe[1]);
		char* arg1[1024];
		tokenizing(argv[1],arg1);
		if(execvp(arg1[0],arg1)==-1){
			perror("Wrong command\n");
			exit(-1);		
		}
	}
	else if(child!=-1){
		close(apipe[1]);
		dup2(apipe[0],fileno(stdin));
		close(apipe[0]);
		int stat_val;
		pid_t child_pid;
		child_pid = wait(&stat_val);
		char* arg2[1024];
		tokenizing(argv[2],arg2);
		if(execvp(arg2[0],arg2)==-1){
			perror("Wrong command\n");
			exit(-1);
		}
	}
	else{
		perror("cannot fork\n");
		exit(-1);
	}
	return 0;
}
void tokenizing(char* line,char** args) {
    args[0]=NULL;
    if (line==NULL) {
        printf("No line\n");
	exit(-1);
    }
    else{
            int i = 0;
            char* pl = line;
            char * str = malloc(sizeof(char)*1024);
            char* pstr = str;
            while(*pl!='\0'){
                if(*pl!=' '){
                    *pstr = *pl;
                    pl++;
                    pstr++;
                }
                else{
                    *pstr = '\0';
                    args[i]=malloc(sizeof(char)*strlen(str));
                    memcpy(args[i], str, strlen(str));
                    pstr = str;
                    i++;
                    pl++;
                    while(*pl==' '){
                        pl++;
                    }
                }
            }
            if(pstr!=str){
                *pstr = '\0';
                args[i]=malloc(sizeof(char)*strlen(str));
                memcpy(args[i], str, strlen(str));
                pstr = str;
                pl++;
                i++;
            }
	    args[i]=NULL;
    }
}


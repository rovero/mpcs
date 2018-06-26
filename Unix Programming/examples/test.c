#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>

int main(int argc,char** argv){
    int fd1[2];//child writes,parent reads
    int fd2[2];//parent writes,child reads
    char* str1 = "one\n";
    char* str2 = "two\n";
    char* str3 = "three\n";
    char* str4 = "four\n";
    char* str5 = "five\n";
    pid_t child;
    static char message[BUFSIZ];
    if(pipe(fd)==-1){
        printf("pipe\n");
        eixt(-1);
    }
    if((child=fork())==0){
        close(fd1[0]);
        close(fd2[1]);
        dup2(fd2[0],fileno(stderr));
        write(fd1[1],str1,len(str1));
        write(fd1[1],str2,len(str2));
        write(fd1[1],str3,len(str3));
        write(fd1[1],str4,len(str4));
        write(fd1[1],str5,len(str5));
    }
    else{
        close(fd1[1]);
        close(fd2[0]);
        dup2(fd1[0],fileno(stdout));
        
    }
}

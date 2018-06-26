#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#define MAX_SIZE 128
int main(int argc, char**argv)
{
    if(argc!=3){
        perror("there should be three arguments\n");
        exit(EXIT_FAILURE);
    }
    
    if(strcmp(argv[1],argv[2])==0){
        perror("The source file and the target file should not be the same one\n");
        exit(1);
    }
    int source, target;
    source = open(argv[1],O_RDWR|O_CREAT);
    if(source==-1){
        perror("The file does not exist\n");
        exit(EXIT_FAILURE);
    }
    
    
    target = open(argv[2],O_CREAT|O_WRONLY);
    if(target==-1){
        perror("fail to create\n");
        exit(EXIT_FAILURE);
    }
    
    
    char source_content[MAX_SIZE];
    int len = 0;
    while((len=read(source,source_content,MAX_SIZE))!=0){
        printf("%s\n",source_content);
        if(write(ptarget,source_content,len)==-1){
            perror("fail to write\n");
            exit(EXIT_FAILURE);
        }
    }
    
    
    close(target);
    close(source);
 
   return 0;
}


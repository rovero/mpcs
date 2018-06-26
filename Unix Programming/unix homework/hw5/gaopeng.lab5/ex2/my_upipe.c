#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>

#define CHILD (pid_t) 0

int main(int argc, char * argv[] )
{
    pid_t child;
    int fdes[2];
    static char message[BUFSIZ];
    
    if(argc!=3){
        perror("argc should be 3\n");
        exit(1);
    }
    char *first[256], *second[256];
    /*deal with the first command*/
    char * pch;
    int count=0;
    pch = strtok (argv[1]," ");
    while (pch != NULL)
    {
        
        first[count++] = pch;
        pch = strtok (NULL, " ");
    }
    first[count] = NULL;
    count = 0;
    /*deal with the second command*/
    pch = strtok (argv[2]," ");
    while (pch != NULL)
    {
        
        second[count++] = pch;
        pch = strtok (NULL, " ");
    }
    second[count] = NULL;
    
    if( pipe(fdes) == -1 )
    {
        perror("Pipe");
        exit(-1);
    }
    
    if( (child = fork()) == CHILD )
    {
        dup2(fdes[1], fileno(stdout));
        close(fdes[0]);
        close(fdes[1]);
        
        execvp(first[0],first);
        exit(5);
    }
    else
    {
        dup2(fdes[0], fileno(stdin));
        close(fdes[0]);
        close(fdes[1]);
        execvp(second[0],second);
        exit(6);
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <string.h>
#include "gosh.h"

extern int errno;





int cd_fork_command(struct command_t *cmd);
int set_environment_variable(struct command_t *cmd);
int get_envir(struct command_t *cmd);

void ouch(int sig) {
    int stat_val;
    pid_t child_pid;
    child_pid = wait(&stat_val);
    printf("a child process: %d has been reaped\n", child_pid);
}


int main() {
    
    struct sigaction act;
    act.sa_handler = ouch;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGCHLD, &act, 0);
    
    
    int retval;
    struct command_t command_a;
    /* Problem 3
     retval = start_sig_catchers();
     */
    
    while(1) {
        retval = init_command(&command_a);
        printf("gosh> ");
        retval = simple_accept_input(&command_a);
        print_command(&command_a, "c1");
        /*
         If the input processing was sucessful, call a fork/exec
         function.  Otherwise, exit the program
         */
        if (retval == 0) {
            char *cd = "cd";
            char *pwd = "pwd";
            char *echo = "echo";
            char *dot="=";
            // Problem 2
            printf("Running Command\n");
            printf("--------------- \n");
            if(strcmp(command_a.args[0],cd) == 0){
                if(cd_fork_command(&command_a)==0)
                    printf("change dir: success\n");
                else
                    printf("change dir: not success\n");
            }
            else if(command_a.num_args==1&& strstr(command_a.args[0], dot) != NULL){
                set_environment_variable(&command_a);
            }
            else if(strcmp(command_a.args[0],echo) == 0 ){
                get_envir(&command_a);
            }
            else{
                simple_fork_command(&command_a);
            }
            printf("--------------- \n");
            printf("Command Returned Exit Code 0\n");
            
        } else if (retval == 1) {
            exit(0);
        }
        
    }
    exit(0);
}

/* Helper Function.  Initialized a command_t struct */
int init_command(struct command_t *cmd) {
    cmd->num_args = 0;
    cmd->args[0] = NULL;
    cmd->outfile[0] = '\0';
    cmd->infile[0] = '\0';
    return(0);
}

/* Helper Function.  Print out relevent info stored in a command_t struct */
int print_command(struct command_t *cmd, const char *tag) {
    int i;
    
    for (i=0; i<cmd->num_args; i++) {
        printf("%s %d: %s\n", tag, i, cmd->args[i]);
    }
    
    if (cmd->outfile[0] != '\0') {
        printf("%s outfile: %s\n", tag, cmd->outfile);
    }
    
    if (cmd->infile[0] != '\0') {
        printf("%s infile: %s\n", tag, cmd->infile);
    }
    
    return(0);
}


/* Problem 1: read input from stdin and split it up by " " characters.
 Store the pieces in the passed in cmd_a->args[] array.  If the user
 inputs 'exit', return a 1.  If the user inputs nothing (\n), return
 a value > 1.  If the user inputs somthing else, return a 0. */
int simple_accept_input(struct command_t *cmd_a) {
    size_t size;
    char *line = NULL;
    char * pch = NULL;
    if (getline(&line, &size, stdin) == -1) {
        printf("No line\n");
    } else {
        char *ll;
        ll = (char*)malloc(strlen(line));
        memset(ll, 0, strlen(line));
        memcpy(ll, line, strlen(line));
        //pch = strtok (ll," ,.");
        char *nothing = "\n";
        
        char *exit = "exit";
        if(strcmp(ll,exit)==0)
            return 1;
        if(strcmp(ll,nothing)==0)
            return 2;
        
        //  printf("%s\n", line);
        
        char *dot="=";
        
        int count = 0;
        if(strstr(line, dot) != NULL){
            //cmd_a->args[count++]="export";
            cmd_a->args[count++] = line;
            cmd_a->args[count] = NULL;
            cmd_a->num_args = count;
            char *first;
            first = strchr(line, '=');
            int index = (int)(first-line);
            char *des=(char*)malloc(strlen(first)-3);
            memset(des, 0, strlen(first)-3);
            memcpy(des, first+2, strlen(first)-4);
            char c = '\n';
            memcpy(des+strlen(first)-4, &c, 1);
            char *subres;
            subres = (char*)malloc(strlen(line)-2);
            memset(subres, 0, strlen(line)-2);
            memcpy(subres, line, index+1);
            memcpy(subres+index+1,des, strlen(first)-3);
            
            printf("%s\n",subres);
            cmd_a->args[++count] = subres;
            return 0;
            
        }
        line[strlen(line)-1]='\0';
        pch = strtok (line," ,.");
        printf("%s\n",pch);
        
        while (pch != NULL)
        {
            //   printf ("~~%s\n",pch);
            cmd_a->args[count++] = pch;
            pch = strtok (NULL, " ,.");
        }
        int i=0;
        /*   for( i=0;i<count;i++){
         printf("%s\n",cmd_a->args[i]);
         }*/
        cmd_a->args[count] = NULL;
        cmd_a->num_args = count;
        
    }
    
    
    return(0);
}




int set_environment_variable(struct command_t *cmd) {
    
    
    pid_t child;
    //printf("parent process id is: %d\n",getpid());
    
    if( (child = fork()) == 0 )
    {
        printf("child:  in child, my process id is %d\n",getpid());
        int temp = cmd->num_args+1;
        printf("%d\n",temp);
        printf("%s\n",cmd->args[temp]);
        if(putenv(cmd->args[temp]) != 0){
            perror("wrong command\n");
            exit(-1);
        }
        
        
        //printf("child:  my parents process id is %d\n",getppid());
    }
    else
    {
        //printf("parent:  still in parent, my process id is still %d\n",getpid());
        //printf("parent:  and my child is %d\n",child);
        sleep(1);
    }
    return 0;
    
}
//putenv



int cd_fork_command(struct command_t *cmd) {
    
    if(chdir(cmd->args[1])==0)
        return 0;
    else{
        return -1;
        
    }
    
    
}


int get_envir(struct command_t *cmd) {
    
    pid_t child;
    printf("int get_envir function\n");
    if( (child = fork()) == 0 )
    {
        printf("!!!!!!%s\n",cmd->args[1]+1);
        printf("%s",getenv(cmd->args[1]+1));
    }
    else
    {
        sleep(1);
    }
    return 0;
    
}



/* Problem 2: write a simple fork/exec/wait procedure that executes
 the command described in the passed in 'cmd' pointer. */
int simple_fork_command(struct command_t *cmd) {
    pid_t child;
    if( (child = fork()) == 0 )
    {
        if(execvp(cmd->args[0],cmd->args) == -1){
            perror("wrong command\n");
            exit(-1);
        }
    }
    else
    {
        sleep(1);
    }
    return(0);
}

/* Problem 3: set up all of your signal actions here */
int start_sig_catchers(void) {
    return(0);
}

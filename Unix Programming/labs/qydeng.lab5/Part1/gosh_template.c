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
void sigchldhandler(int sig){
	int stat_val;
	pid_t child_pid;
	child_pid = wait(&stat_val);
	printf("child %d has been reaped\n",child_pid);
}
int main() {
    struct sigaction action;
    action.sa_handler = sigchldhandler;
    action.sa_flags=0;
    sigemptyset(&action.sa_mask);
    sigaction(SIGCHLD,&action,0);

    int retval;
    struct command_t command_a;
    

     retval = start_sig_catchers();
    
    while(1) {
        
        retval = init_command(&command_a);
        
        printf("gosh> ");
        /* this getchar() function is just here so that the shell 'works'
         without any modification.  comment this function call out
         before you start working on either of the two funtions below
         for input processing */
        //getchar();
        // Problem 1
        retval = simple_accept_input(&command_a);
        //user type exit
        print_command(&command_a, "c1");
        
        /*
         If the input processing was sucessful, call a fork/exec
         function.  Otherwise, exit the program
         */
        if (retval == 1) {
            exit(0);
        }
        else if (retval == 0) {
            printf("%s\n","Running Command");
            printf("%s\n","---------------");
            if(strcmp(command_a.args[0],"cd")==0){
                if(chdir(command_a.args[1])==0){
                    printf("change dir: success\n");
                }
                else{
                    printf("change dir: not success\n");
                }
            }
            else if(command_a.num_args==1&&strstr(command_a.args[0],"=")!=NULL){
                set_env(command_a.args[0]);
            }
            else if(command_a.num_args==2&&strcmp(command_a.args[0],"echo")==0){
                get_env(&command_a);
            }
	    else if(command_a.num_args==4&&strcmp(command_a.args[3],"2>&1")==0){
		redirect(&command_a);
	    }
            else{
                simple_fork_command(&command_a);
            }
            printf("--------------- \n");
            printf("Command Returned Exit Code 0\n");
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
    char *line = NULL;
    size_t size;
    int flag = 0;//0 outside quotation, 1 inside quotation
    if (getline(&line, &size, stdin) == -1) {
        printf("No line\n");
    }
    else{
        if(strcmp(line,"\n")==0){
            return 2;
        }
        else if(strcmp(line,"exit\n")==0){
            return 1;
        }
        else{
            int i = 0;
            char* pl = line;
            char * str = malloc(sizeof(char)*1024);
            char* pstr = str;
            //printf("%c\n",*pl);
            while(*pl!='\n'){
                if(*pl!=' '){
                    //printf("a%c\n",*pl);
                    *pstr = *pl;
                    pl++;
                    pstr++;
                    if(*pl=='\''){
                        if(flag==0){
                            flag=1;//enter quotation mark
                        }
                        else{
                            flag=0;//exit quotation mark
                        }
                    }
                }
                else{ //*pl=' '
                    if(flag==0){
                        *pstr = '\0';
                        //printf("b%c\n",*pl);
                        //printf("%s\n",str);
			/*if(i>0&&strcmp(cmd_a->args[i-1],"<")==0){
				//memset(cmd_a->infile, '\0', sizeof(cmd_a->infile));
				strcpy(cmd_a->infile, str);
				pstr = str;
                        	pl++;			
			}
			if(i>0&&strcmp(cmd_a->args[i-1],">")==0){
				//memset(cmd_a->outfile, '\0', sizeof(cmd_a->outfile));			
				strcpy(cmd_a->outfile, str);
				pstr = str;
                        	pl++;
			}
			else{*/
				if(i>0&&strcmp(cmd_a->args[i-1],">")==0){
					strcpy(cmd_a->outfile, str);				
				}
                        	cmd_a->args[i]=malloc(sizeof(char)*strlen(str));
                        	memcpy(cmd_a->args[i], str, strlen(str));
                        	pstr = str;
                        	i++;
                        	pl++;
			//}
                        while(*pl==' '){
                            pl++;
                        }
                    }
                    else{
                        *pstr = *pl;
                        pl++;
                        pstr++;
                    }
                }
            }
            if(pstr!=str){
                *pstr = '\0';
                cmd_a->args[i]=malloc(sizeof(char)*strlen(str));
                memcpy(cmd_a->args[i], str, strlen(str));
                pstr = str;
                pl++;
                i++;
            }
            cmd_a->args[i]=NULL;
            cmd_a->num_args=i;
            return 0;
        }
    }
    return(0);
}
int redirect(struct command_t *cmd){
	pid_t pid;
	int fd;
	//printf("start\n");
	if((pid=fork())==-1){
		perror("fork");exit(1);
	}
	if(pid==0){
		close(1);
		//printf("outfile %s\n",cmd->outfile);
		fd=creat(cmd->outfile,0644);
		dup2(fd,2);
		char * tmp = strdup(cmd->args[0]);
		char * add = malloc(sizeof(char)*(2+strlen(cmd->args[0])));
		strcpy(add,"./");		
		strcat(add,tmp);
		//printf("%s\n",add);
		if(execlp(add,add,NULL)==-1){
			perror("Command does not exist\n");		
			exit(-1);	
		}
			
	}
	if(pid!=0){
		int stat_val;
        	pid_t child_pid;
        	child_pid = wait(&stat_val);
		if(stat_val!=-1){
			printf("dup file handle: success\n");		
		}
	}
    
    return(0);
	
}
/* Problem 2: write a simple fork/exec/wait procedure that executes
 the command described in the passed in 'cmd' pointer. */
int simple_fork_command(struct command_t *cmd) {
    //int flag=0;
    pid_t new_pid;
    new_pid = fork();
    if(new_pid==0){
        if(execvp(cmd->args[0],cmd->args)==-1){
            //flag=1;
            perror("Command does not exist\n");
            exit(-1);
        }
    }
    else{
        int stat_val;
        pid_t child_pid;
        child_pid = wait(&stat_val);
    }
    
    return(0);
}
int set_env(char* line){
    pid_t child;
    //if((child=fork())==0){
        if(putenv(line)!=0){
            perror("Command does not exist\n");
            exit(-1);
        }
        else{
            printf("define new variable: success\n");
        }
    //}
    //else{
     //   int stat_val;
     //   pid_t child_pid;
     //   child_pid = wait(&stat_val);
    //}
    return 0;
}
int get_env(struct command_t * cmd){
    pid_t child;
    char* path;
    //if((child=fork())==0){
        if(*(cmd->args[1])=='$'&&(path=getenv(cmd->args[1]+1))!=NULL){
            char dest[strlen(path)-1];
            memset(dest, '\0', sizeof(dest));
            strncpy(dest,path+1,strlen(path)-2);
            printf("%s\n",dest);
            //exit(0);
        }
        else{
            printf("%s\n","Variable does not exist");
            exit(-1);
        }
    //}
    //else{
    //    int stat_val;
    //    pid_t child_pid;
    //    child_pid = wait(&stat_val);
    //}
    return 0;
}
/* Problem 3: set up all of your signal actions here */
int start_sig_catchers(void) {
    return(0);
}

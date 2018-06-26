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
#include <gosh.h>

extern int errno;






main() {

  int retval;
  
  struct command_t command_a;

  /* Problem 3
     retval = start_sig_catchers();
  */

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
    

    print_command(&command_a, "c1");

    /* 
       If the input processing was sucessful, call a fork/exec
       function.  Otherwise, exit the program 
    */
    if (retval == 0) {      

      // Problem 2
        printf("Running Command\n");
        printf("--------------- \n");
        simple_fork_command(&command_a);
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
    char * pch;
    if (getline(&line, &size, stdin) == -1) {
        printf("No line\n");
    } else {
      //  printf("%s\n", line);
        line[strlen(line)-1]='\0';
        char * pch;
        pch = strtok (line," ,.");
        char *exit = "exit";
        char *nothing = "n";
        if(strcmp(pch,exit)==0)
            return 1;
        if(strcmp(pch,nothing)==0)
            return 2;
        int count = 0;
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

/* Problem 2: write a simple fork/exec/wait procedure that executes
   the command described in the passed in 'cmd' pointer. */
int simple_fork_command(struct command_t *cmd) {
    
    pid_t child;
    //printf("parent process id is: %d\n",getpid());
    
    if( (child = fork()) == 0 )
    {
        //printf("child:  in child, my process id is %d\n",getpid());
        if(execvp(cmd->args[0],cmd->args) == -1){
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
    
    
    
    
  return(0);
}

/* Problem 3: set up all of your signal actions here */
int start_sig_catchers(void) {
  return(0);
}

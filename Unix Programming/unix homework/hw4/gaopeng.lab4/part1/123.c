#include <sys/types.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
static int MAXSTOPS = 0;
static int count =0;


void sig_int_handler(int sig)
{
  /* unreliable signals used to have to RESET signal(SIGUSR1,handler) here */
    count++;
    char message[] = "You've pressed Ctrl-C 1 times.  Aren't you getting the message that I'm invulnerable?\n";
    message[22] = count+'0';
    if(count<MAXSTOPS){
        psignal(sig, message);
    }
    else{
        exit(-1);
    }
}


void sig_usr1_handler(int sig)
{
    /* unreliable signals used to have to RESET signal(SIGUSR1,handler) here */
        psignal(sig, "receive SIGUSR1\n");
}
void ouch(int sig)
{
    printf("OUCH!  I got a signal: %d\n",sig);
}



int main(int argc,char **argv)
{
  void * retval;
  MAXSTOPS = (*(argv[1]))-'0';
  //printf("process id: %d\n",getpid());
  if( (retval = signal(SIGINT,sig_int_handler)) == SIG_ERR)
  {
    fprintf(stderr, "Oops, failed to establish handler for SIGINT\n");
    exit(-1);
  }
  if( (retval = signal(SIGUSR1,sig_usr1_handler)) == SIG_ERR)
  {
    fprintf(stderr, "Oops, failed to establish handler for SIGUSR1\n");
    exit(-1);
  }
    while(1){
    //printf("hi~~~\n");  /* block until signal is received */
    sleep(3);
    }
}



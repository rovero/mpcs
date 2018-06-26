#include <sys/types.h>
#include <stdio.h>
#include <signal.h>

void handler(int sig)
{
  /* unreliable signals used to have to RESET signal(SIGUSR1,handler) here */
  psignal(sig, "10-4 Little Buddy, got a signal\n");
}

int main(void)
{
  void * retval;
  printf("process id: %d\n",getpid());
  if( (retval = signal(SIGUSR1,handler)) == SIG_ERR)
  {
    fprintf(stderr, "Oops, failed to establish handler for SIGUSR1\n");
    exit(-1);
  }
  if( (retval = signal(SIGUSR2,handler)) == SIG_ERR)
  {
    fprintf(stderr, "Oops, failed to establish handler for SIGUSR2\n");
    exit(-1);
  }
  while(1)
    pause();  /* block until signal is received */
}



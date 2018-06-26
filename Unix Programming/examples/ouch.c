#include <signal.h>
#include <stdio.h>
#include <unistd.h>

void ouch(int sig)
{
        printf("OUCH!  I got a signal: %d\n",sig);
}

int main(void)
{
        (void) signal(SIGINT, ouch);
        while(1)
        {
                printf("Hello World\n");
                sleep(1);
        }
}

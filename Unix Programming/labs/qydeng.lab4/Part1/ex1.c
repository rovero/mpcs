//
//  ex1.c
//  lab4
//
//  Created by Qianyu Deng on 2017/1/31.
//  Copyright © 2017年 Qianyu Deng. All rights reserved.
//

#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

static int counter = 0;
static int Maxstops = 0;
void sigint_handler(){
    counter++;
    if(counter<Maxstops){
        printf("You've pressed Ctrl-C %d times.  Aren't you getting the message that I'm invulnerable?\n",counter);
    }
    else{
        exit(-1);
    }
    
}
void sigusr1_handler(int sig){
    psignal(sig, "signalusr1 received");
}
int main(int argc, char** argv){
    void * retval;
    if((retval=(signal(SIGINT,sigint_handler)))==SIG_ERR){
        fprintf(stderr, "Oops, failed to establish handler for SIGINT\n");
        exit(-1);
    }
    if((retval=(signal(SIGUSR1,sigusr1_handler)))==SIG_ERR){
        fprintf(stderr, "Oops, failed to establish handler for SIGINT\n");
        exit(-1);
    }
    if(argc!=2){
        perror("Usage: ./mysig Maxstops\n");
        exit(-1);
    }
    Maxstops = atoi(argv[1]);
    while(1){
        sleep(1);
    }
    
}

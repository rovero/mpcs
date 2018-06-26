#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/*
* Shared memory server.
*/
#define LEN 1000
int main(int argc, char *argv[])
{
  int shmid;
  char *shm;
  key_t key = IPC_PRIVATE;
  int flag = SHM_R | SHM_W;

  if (argc >= 2) {
    key = atoi(argv[1]);
  }

  shmid = shmget(key, LEN, flag);
  printf("shared memory for key %d: %d\n", key, shmid);
  if (shmid < 0) {
    perror("shmget");
    printf("Try to create this segment\n");
    shmid = shmget(key, LEN, flag | IPC_CREAT);
    if (shmid < 0) {
      perror("shmget | IPC_CREAT");
    }
  }

  shm = (char *)shmat(shmid, /*addr*/0, /*flag*/0); 
  if (shm == (char *)-1) {
    perror("shmat");
    exit(1);
  }
    
  sprintf(shm, "1");
  printf("shared memory content: %s\n", shm);
  //char *shm;
    shm = (char *)shmat(shmid, /*addr*/0, /*flag*/0);
    while(1){
        
        int num = atoi(shm);
        printf("num = %d\n",num);
        if(num>=5){
            sleep(2);
            continue;
        }
        num++;
        char c[10];
        sprintf(c,"%d",num);
        printf("------%s\n",c);
        sprintf(shm,c);
        sleep(2);
    }
    
    
    
    
  return 0;
} /* main */

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sem.h>
union semun {
    int val;
    struct semid_ds *buf;
    unsigned short int *array;
};
/*
* Shared memory client.
*/

static int sem_id;

static int semaphore_p(void) {
    struct sembuf sem_b;
    sem_b.sem_num = 0;
    sem_b.sem_op = -1; /* P() */
    sem_b.sem_flg = SEM_UNDO;
    if (semop(sem_id, &sem_b, 1) == -1) {
        fprintf(stderr,"semaphore_p failed\n");
        return (0);
    }
    return 1;
}


static int semaphore_v(void) {
    struct sembuf sem_b;
    sem_b.sem_num = 0;
    sem_b.sem_op = 1; /* V() */ sem_b.sem_flg = SEM_UNDO;
    if (semop(sem_id, &sem_b, 1) == -1) {
        fprintf(stderr, "semaphore_v failed\n");
        return(0);
    }
    return(1);
}

static void del_semvalue(void) {
    union semun sem_union;
    if (semctl(sem_id, 0, IPC_RMID, sem_union) == -1) fprintf(stderr, "Failed to delete semaphore\n");
}



static int set_semvalue(void) {
    union semun sem_union;
    sem_union.val = 1;
    if (semctl(sem_id, 0, SETVAL, sem_union) == -1) return(0); return(1);
}
int main(int argc, char *argv[])
{
  int shmid;
  char *shm;
  key_t key;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s key\n", argv[0]);
    exit(2);
  }
  key = atoi(argv[1]);

  shmid = shmget(key, 10, SHM_R | SHM_W|IPC_CREAT);
  if (shmid < 0) {
    perror("shmget");
    shmid = key;
  }
  else {
    printf("shared memory: %d\n", shmid);
  } 

  shm = (char *)shmat(shmid, /*addr*/0, /*flag*/0); 
  if (shm == (char *)-1) {
    perror("shmat");
    exit(1);
  }
  if (shm != 0) {
    printf("shared memory content: %s\n", shm);
  }
  
    sem_id = semget((key_t)1236, 1, 0666 | IPC_CREAT);
    
    while(1){
        //shm = (char *)shmat(shmid, /*addr*/0, /*flag*/0);
        if (!semaphore_p()) exit(EXIT_FAILURE);
        
        int num = atoi(shm);
        num--;
        printf("One client takes one, now left: %d\n",num);
        char c[10]={0};
        
        sprintf(c,"%d",num);
        sprintf(shm,c);
        

        /* END of CRITICAL SECTION */
        if (!semaphore_v()) exit(EXIT_FAILURE);
        sleep(1);
        
    }
    
  return 0;
} /* main */

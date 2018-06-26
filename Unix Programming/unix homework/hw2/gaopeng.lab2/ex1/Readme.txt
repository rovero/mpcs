The program support the basic operation and the second option (%a %A %g %i %b)





Script started on Thu 21 Jan 2016 12:32:20 PM CST
gaopeng@linux2:~/hw1$ ./mystat p1
"File: "p1
Size: 8600		Blocks: 24	I/O Block:16384	  Regular File
Device: 1ah/26d	      Inode: 2178128561      Link: 1
Access:(0770/-rwxrwx---)  Uid: (13841/ gaopeng)   Gid: (17124/ gaopeng)
Access: Tue Sep 29 21:19:36 2015
Modify: Mon Oct  5 12:24:31 2015
Change: Tue Sep 29 21:19:36 2015
gaopeng@linux2:~/hw1$ ./mystat -c -format=%A p1
-format=-rwxrwx---
gaopeng@linux2:~/hw1$ exit
exit

Script done on Thu 21 Jan 2016 12:32:51 PM CST
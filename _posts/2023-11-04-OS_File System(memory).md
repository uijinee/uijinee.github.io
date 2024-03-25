---
title: "4. File System(memory)"
date: 2023-11-04 22:00:00 +0900
categories: ["Computer Science", "Operating System"]
tags: ["os", "operating system"]
use_math: true
---

앞서 우리는 On-Disk File System에 대해서 알아보았다.

이 On-Disk File System은 **Inode**를 통해 Time Efficiency를 효과적으로 늘렸지만 다음과 같은 단점이 여전히 존재했다.
- **여전히 느리다.** <br>
- **여러 Disk의 관리가 힘들다.**<br>
    *(System이 여러개의 Disk를 가지면 각 Disk마다 File System이 있어야 함)*

따라서 On-Memory File System(Virtual File System)이라는 방법을 통해 위의 두 문제를 해결했는데, 주된 Concept은 다음과 같다.
- 속도향상 &rarr; <u>**Caching**</u><br>
    : 자주 접근하는 Block은 System Memory에 복사해 놓고 사용
- 여러 Disk 관리 &rarr; <u>**Mounting**</u><br>
    : File System에 File System을 연결하여 하나의 File System을 통해 여러개의 File System을 관리하는 방법

---
# On-Memory File System

## Caching

Caching은 자주 접근하는 정보를 미리 System Memory에 저장해 놓고 필요할때마다 사용하는 것으로, Disk에 매번 접근할 필요가 없기 때문에 속도향상에 매우 큰 영향을 준다.

이때 Cache의 대상이 되는 것은 다음과 같다.
- Super Block
- Group Descriptor
- DBM(Data Block Bitmap)
- IBM(Inode Bitmap)

이때, System Memory는 해당 Block이 어느 Disk의 Block인지에 대한 정보 등을 같이 알고 있어야 하기 때문에 원래 데이터에 Additional Data를 추가하고, <br>
이렇게 Cache된 Data를 Linked List에 연결하여 보관한다.

### 1) Super Block

![Alt text](/assets/img/post/operating_system/onmemory_superblock.png)

> ```cpp
> /* include/linux/fs.h */
> struct super_block {
>     dev_t               s_dev;      /* 이 superblock을 소유하는 device # */
>     char                s_id[32];   /* 이 superblock을 소유하는 device name */
>     struct dentry       *s_root;    /* 이 file system의 root directory 위치*/
>     struct list_head    s_files;    /* 이 file system에 속한 file들의 linked list */    
>     struct list_head    s_list;     /* next super block의 위치 */
>     unsigned long       s_blocksize;
>     ...
> };
> ```
> 
> ---
> #### Define
> 
> On-Memory Super Block은 Mount된 On-Disk File System에서 다음의 정보를 결합하여 새로운 super block을 만든다.<br>
>    *(`include/linux/fs.h`에 정의되어 있다.)*
> - `ext2_super_block`
> - `Additional info`
>
>> 이때 이 super block은 `fs/super.c`에 선언된 `super_blocks`라는 linked list에 연결되어 cache된다.
> ---
> #### Example
> 
> Cache된 모든 Super Block을 출력하기 
> ```cpp
> void display_super_blocks(void) {
>     struct super_block *sb;
>     list_for_each_entry(sb, &super_blocks, s_list) {
>         printk("dev name: %s, dev major num: %d, dev minor num: %d, root ino: %d\n",
>                sb->s_id, MAJOR(sb->s_dev), MINOR(sb->s_dev), sb->s_root->d_inode->i_ino);
>     }
> }
> ```
>
> - `list_for_each_entry([정보를 담을 변수, linked list, next pointer])`<br>
> - `MAJOR(): Device 번호`<br>
> - `MINOR(): 그 Device의 종류안에서 구별할 수 있는 번호`

### 2) Inode

![Alt text](/assets/img/post/operating_system/onmemory_inode.png)

> ```cpp
> /* include/linux/fs.h */
> struct inode {
>     dev_t               i_rdev      /* 이 inode를 소유하는 device # */
>     unsigned long       i_ino;      /* inode number */
>     struct super_block  *i_sb;      /* 이 inode가 속하는 superblock의 위치 */
>     struct list_head    i_dentry;   /* 이 inode를 관리하기위한 dentry list*/
>     struct list_head    i_list;     /* next_inode의 위치 */
>     ...
> };
> 
> /* include/linux/deache.h */
> struct dentry {
>     struct inode    *d_inode;   /* 이 dentry가 속한 inode에 대한 pointer */
>     struct qstr     d_name;     /* corresponding file name (->d_name.name) */
>     int             d_mounted;  /* 이 inode가 mounting point인지 */
>     ...
> };
> ```
>
> ---
> #### Define
>
> On-Memory Inode는 Mount된 On-Disk File System에서 Inode의 정보와 다음의 정보를 결합하여 새로운 super block을 만든다.<br>
>    *(`include/linux/fs.h`에 정의되어 있다.)*
> - `ext2_inode`
> - `Additional info`
> - `dentry`
>
>> 이때 inode는 `fs/inode.c`에 선언된 `inode_in_use`라는 linked list에 연결되어 cache된다.
> 
> *(dentry: Directory Entry의 약자로, inode를 효율적으로 관리하기 위한 자료구조)*
>
> ---
> #### Example
>
> Cache된 모든 Inode 출력하기
> ```cpp
> extern struct list_head inode_in_use;
> void display_all_inodes(){
>     struct inode *in;
>     struct dentry *den;
>     list_for_each_entry(in, &inode_in_use, i_list){
>            printk("dev maj num:%d dev minor num:%d inode num:%d sb dev:%s\n", 
>                   MAJOR(in->i_rdev), MINOR(in->i_rdev), in->i_ino, in->i_sb->s_id);
>            list_for_each_entry(den, &in_i->dentry, d_alias){
>                   printk("file name: %s, file byte size: %x\n",
>                           den->d_name.name)
>            }
>     } 
> }
> ```

### 3) 그 밖의 Block들

> 위의 두 자료구조 외에도 자주 사용하는 정보(Block)들도 Block단위로 따로 Cache해둔다.
>
> 이때 사용하는 구조체는 다음과 같다.
> ```cpp
> /* include/linux/buffer_head.h */
> struct buffer_head {
>     struct block_device *b_bdev;    /* 현재 이 Block을 소유하고 있는 device #*/
>     sector_t            b_blocknr;  /* 이 Block의 Block #*/
>     size_t              b_size;
>     ...
> }
> ```
>
>> 이때 이 Block들은 최대한 빨리 접근하는 것이 중요하기 때문에,<br>
>> linked list가 아닌 <u>**hash table**</u>을 사용해서 Cache한다. 
> 
> *(linux 2.6이상부터는 hash table도 아닌 더 복잡한 자료구조를 사용해 Cache한다.)*

---
## Mounting

![Alt text](/assets/img/post/operating_system/mounting.png)


### 1) Concept

> #### Root File System
>
> System Memory에 가장 처음 Cache되는 File System으로 다른 On Disk File System이 여기에 mount된다.
> #### mounted file system
> #### mounting point

### 2) 과정

> ![Alt text](/assets/img/post/operating_system/mount_process(1).png)
> 
> #### 1. Cache the Mounted File System
>
> ![Alt text](/assets/img/post/operating_system/mount_process(2).png)
>
> Mounted File System을 Cache하여 System Memory로 가지고 온다.<br>
> 이때, Cache하는 정보는 다음 두가지이다.
>
> - superblock
> - root inode<br>
>   *(Mounted File System의 inode 중 root inode만을 Cache한다.)*
>
> ---
> #### 2. Cache the Mounting Point's Inode
> 
> ![Alt text](/assets/img/post/operating_system/mount_process(3).png)
>
> 다음으로는 Mounted File System을 연결할 Mounting Point를 가져와야 한다.<br>
> 이를 위해서 Mounting Point의 Inode를 Cache한다.
>
> *(즉, `'/'(root)`부터 Mounting Point까지 경로상의 모든 dir에 대한 inode를 Cache해야 한다.)*
>
> ---
> #### 3. Mounted File System을 Mounting Point에 연결
> 
> ![Alt text](/assets/img/post/operating_system/mount_process(4).png)
> 
> 연결 과정은 다음과 같다.<br>
>
> **a)** Mounting Point가 되는 File의 Inode를 찾아 그 dentry의 d_mounted += 1
>
> **b)** `vfsmount{}`자료구조를 통해 연결정보를 저장
>    - mnt_mountpoint<br>
>       : mounting point위치 저장
>    - mnt_root<br>
>       : Mounted File System의 root inode 저장 
>    - mnt_sb<br>
>       : Mounted File System의 super block 저장
>
> **c)** `vfsmount{}`자료구조를 mount_hastable에 저장


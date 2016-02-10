<?php

// Connection parameters
$host = 'cspp53001.cs.uchicago.edu';
$username = 'qydeng';
$password = 'TEST';
$database = $username.'DB';

// Attempting to connect
$dbcon = mysqli_connect($host, $username, $password, $database)
   or die('Could not connect: ' . mysqli_connect_error());
print 'Connected successfully!<br>';

// Getting the input parameter (user):
$id = $_REQUEST['id'];
$opt = $_REQUEST['option'];
$info = $_REQUEST['info'];

$sql;
if($opt == 1){
  $sql = "UPDATE Doctor SET name = '$info' WHERE work_id = $id";
}
elseif($opt == 2){
   $sql = "UPDATE Doctor SET gender = '$info' WHERE work_id = $id";
}
elseif($opt == 3){
   $sql = "UPDATE Doctor SET date_of_birth = '$info' WHERE work_id = $id";
}
elseif($opt == 4){
    $depId = (int)$info;
    $sql = "UPDATE Doctor SET department_id = $depId WHERE work_id = $id";
}

if ($dbcon->query($sql) === TRUE) {
    echo "Record updated successfully"."<br>";
    $query = "SELECT work_id, d.name AS doctor_name, gender, date_of_birth, dp.name AS dept_name FROM Doctor d JOIN Department dp on d.department_id = dp.id WHERE work_id = $id";
    $result = mysqli_query($dbcon, $query)
      or die('Query failed: ' . mysqli_error($dbcon));
    // Can also check that there is only one tuple in the result
    $tuple = mysqli_fetch_array($result, MYSQL_ASSOC)
      or die("Doctor Work ID $workId not found!");

    print "Doctor with Work ID <b>$workId</b> has the following attributes:";

    // Printing user attributes in HTML
    print '<ul>';
    print '<li> WorkID: '.$tuple['work_id'];
    print '<li> Name: '.$tuple['doctor_name'];
    print '<li> Gender: '.$tuple['gender'];
    print '<li> Date Of Birth: '.$tuple['date_of_birth'];
    print '<li> Department: '.$tuple['dept_name'];
    print '</ul>';
}
else{
    echo "Error updating record: " . $dbcon->error;
}



// Free result
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>

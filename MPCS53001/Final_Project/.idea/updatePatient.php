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
$id = $__REQUEST['id'];
$opt = $_REQUEST['option'];
$info = $___REQUEST['info'];

$sql=empty;
if($opt == 1){
  $sql = "UPDATE Patient SET first_name = '$info' WHERE id = $id";
}
elseif($opt == 2){
  $sql = "UPDATE Patient SET last_name = '$info' WHERE id = $id";
}
elseif($opt == 3){
  $sql = "UPDATE Patient SET gender = '$info' WHERE id = $id";
}
elseif($opt == 4){
  $date = strtotime('$info');
  $sql = "UPDATE Patient SET date_of_birth = '$date' WHERE id = $id";
}
elseif($opt == 5){
  $dept = int('$info');
  $sql = "UPDATE Patient SET first_name = $dept WHERE id = $id";
}
if ($dbcon->query($sql) === TRUE) {
    echo "Record updated successfully";
}
else{
    echo "Error updating record: " . $dbcon->error;
}

$query = "SELECT * FROM Patient p JOIN Doctor d on d.work_id = p.doctor_work_id WHERE id = $id";
$result = mysqli_query($dbcon, $query)
      or die('Query failed: ' . mysqli_error($dbcon));

    // Can also check that there is only one tuple in the result
$tuple = mysqli_fetch_array($result, MYSQL_ASSOC)
      or die("Patient id $id not found!");

print "Patient with ID <b>$id</b> has the following attributes:";

    // Printing user attributes in HTML
print '<ul>';
print '<li> ID: '.$tuple['id'];
print '<li> First Name: '.$tuple['first_name'];
print '<li> Last Name: '.$tuple['last_name'];
print '<li> Gender: '.$tuple['gender'];
print '<li> Date Of Birth: '.$tuple['date_of_birth'];
print '<li> Attending Doctor: '.$tuple['name'];
print '</ul>';

// Free result
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>

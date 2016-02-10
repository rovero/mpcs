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
$fname = $_REQUEST['fname'];
$lname = $_REQUEST['lname'];
$date = $_REQUEST['dob'];
$sex = $_REQUEST['sex'];
$dept = $_REQUEST['dept'];

$sql = "CALL addDoctor('$sex','$date',$dept,'$fname $lname')";

if ($dbcon->query($sql) === TRUE) {
    echo "Record added successfully<br>";
    $query = "SELECT work_id, gender, date_of_birth,d.name as name,dp.name as dp FROM Doctor d JOIN Department dp on d.department_id = dp.id WHERE work_id =(SELECT max(work_id) FROM Doctor)";
    $result = mysqli_query($dbcon, $query)
      or die('Query failed: ' . mysqli_error($dbcon));

    // Can also check that there is only one tuple in the result
    $tuple = mysqli_fetch_array($result, MYSQL_ASSOC)
      or die("Doctor id $id not found!");



    // Printing user attributes in HTML
    print '<ul>';
    print '<li> ID: '.$tuple['work_id'];
    print '<li> Name: '.$tuple['name'];
    print '<li> Gender: '.$tuple['gender'];
    print '<li> Date Of Birth: '.$tuple['date_of_birth'];
    print '<li> Department: '.$tuple['dp'];
    print '</ul>';
} else {
    echo "Error adding record: " . $dbcon->error;
}
// Get the attributes of the user with the given username


// Free result
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>

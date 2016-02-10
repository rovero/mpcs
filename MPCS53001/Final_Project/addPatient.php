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

$sql = "CALL addPatient('$fname','$lname','$date','$sex',$dept)";

if ($dbcon->query($sql) === TRUE) {
    echo "Record added successfully";
    $query = "SELECT * FROM Patient p JOIN Doctor d on d.work_id = p.doctor_work_id WHERE id =(SELECT max(id) FROM Patient)";
    $result = mysqli_query($dbcon, $query)
      or die('Query failed: ' . mysqli_error($dbcon));

    // Can also check that there is only one tuple in the result
    $tuple = mysqli_fetch_array($result, MYSQL_ASSOC)
      or die("Patient id $id not found!");

    print "Patient <b>$id $ln</b> has the following attributes:";

    // Printing user attributes in HTML
    print '<ul>';
    print '<li> ID: '.$tuple['id'];
    print '<li> First Name: '.$tuple['first_name'];
    print '<li> Last Name: '.$tuple['last_name'];
    print '<li> Gender: '.$tuple['gender'];
    print '<li> Date Of Birth: '.$tuple['date_of_birth'];
    print '<li> Attending Doctor: '.$tuple['name'];
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

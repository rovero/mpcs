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

$query = "SELECT id FROM Hospital_Bed WHERE patient_id = NULL";
$result = mysqli_query($dbcon, $query)
            or die('Query failed: ' . mysqli_error($dbcon));

// Can also check that there is only one tuple in the result
$tuple = mysqli_fetch_array($result, MYSQL_ASSOC)
  or die("No bed is available at this time");

print "These beds are available:";

// Printing user attributes in HTML
print '<ul>';
print '<li> ID: '.$tuple['id'];
print '</ul>';
// Free result
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>

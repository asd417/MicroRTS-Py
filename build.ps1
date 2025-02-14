Set-Location -Path "gym_microrts/microrts"

# Remove existing build directory and microrts.jar if they exist
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, microrts.jar

# Create build directory
New-Item -ItemType Directory -Path build

# Collect all Java source files
$javaFiles = Get-ChildItem -Path ./src -Recurse -Filter *.java | ForEach-Object { $_.FullName }

# Write all file paths to an argument file
$argFile = "sources.txt"
$javaFiles | Set-Content -Path $argFile

# Compile Java files using the argument file
javac -d "./build" -cp "./lib/*;./lib/ejml-v0.42-libs/*" -sourcepath "./src" "@$argFile"

# Copy libraries to build directory
Copy-Item -Path lib\* -Destination build -Recurse

# Remove weka dependency and bots folder
Remove-Item -Force -ErrorAction SilentlyContinue build/weka.jar
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build/bots

# Extract all JAR files and create microrts.jar
Set-Location -Path build

Get-ChildItem -Filter *.jar | ForEach-Object {
    Write-Output "adding dependency $($_.Name)"
    & "$env:JAVA_HOME\bin\jar.exe" xf $_.FullName
}

#jar cvf microrts.jar *
& "$env:JAVA_HOME\bin\jar.exe" cvf microrts.jar *
Move-Item -Path microrts.jar -Destination "../microrts.jar"

# Clean up build directory
Set-Location -Path ..
Remove-Item -Recurse -Force build
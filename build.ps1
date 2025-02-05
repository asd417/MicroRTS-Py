Set-Location -Path "gym_microrts/microrts"

# Remove existing build directory and microrts.jar if they exist
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, microrts.jar

# Create build directory
New-Item -ItemType Directory -Path build

# Compile Java source files
$javaFiles = Get-ChildItem -Path ./src -Recurse -Filter *.java | ForEach-Object { $_.FullName }
javac -d "./build" -cp "./lib/*" -sourcepath "./src" $javaFiles

# Copy libraries to build directory
Copy-Item -Path lib\* -Destination build -Recurse

# Remove weka dependency and bots folder
Remove-Item -Force -ErrorAction SilentlyContinue build/weka.jar
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build/bots

# Extract all JAR files and create microrts.jar
Set-Location -Path build

Get-ChildItem -Filter *.jar | ForEach-Object {
    Write-Output "adding dependency $($_.Name)"
    jar xf $_.FullName
}

jar cvf microrts.jar *
Move-Item -Path microrts.jar -Destination "../microrts.jar"

# Clean up build directory
Set-Location -Path ..
Remove-Item -Recurse -Force build

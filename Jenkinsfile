def checkoutSource(gitCredentialId, organization, repository) {
    withCredentials([[$class: 'UsernamePasswordMultiBinding', credentialsId: gitCredentialId, usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD']]) {
        git url: "https://github.com/${organization}/${repository}.git", branch: env.BRANCH_NAME, credentialsId: gitCredentialId
        sh """
            git config --global user.name '${GIT_USERNAME}'
            git config --global user.email '${GIT_USERNAME}'
            git fetch --tags
        """
    }
}


def isReleaseJob() {
    val tag = sh(returnStdout: true, script: "git tag -l --contains HEAD").trim()
    return tag.startsWith("v")
}

node("JenkinsOnDemand") {
    def repository = 'spark-ml-serving'
    def organization = 'Hydrospheredata'
    def gitCredentialId = 'HydrospheredataGithubAccessKey'

    stage('Checkout') {
        deleteDir()
        checkoutSource(gitCredentialId, organization, repository)
    }

    stage('Test') {
        sh "${env.WORKSPACE}/sbt/sbt -no-colors -J-Xss2m spark_20/test"
        sh "${env.WORKSPACE}/sbt/sbt -no-colors -J-Xss2m spark_21/test"
        sh "${env.WORKSPACE}/sbt/sbt -no-colors -J-Xss2m spark_22/test"
    }

    if (isReleaseJob()) {
        stage("Publish"){
            sh "${env.WORKSPACE}/sbt/sbt -DsparkVersion=${v} 'set pgpPassphrase := Some(Array())' common/publishSigned"
            sh "${env.WORKSPACE}/sbt/sbt -DsparkVersion=${v} 'project common' 'sonatypeRelease'"

            sh "${env.WORKSPACE}/sbt/sbt -DsparkVersion=${v} 'set pgpPassphrase := Some(Array())' spark_20/publishSigned"
            sh "${env.WORKSPACE}/sbt/sbt -DsparkVersion=${v} 'project spark_20' 'sonatypeRelease'"

            sh "${env.WORKSPACE}/sbt/sbt -DsparkVersion=${v} 'set pgpPassphrase := Some(Array())' spark_21/publishSigned"
            sh "${env.WORKSPACE}/sbt/sbt -DsparkVersion=${v} 'project spark_21' 'sonatypeRelease'"

            sh "${env.WORKSPACE}/sbt/sbt -DsparkVersion=${v} 'set pgpPassphrase := Some(Array())' spark_22/publishSigned"
            sh "${env.WORKSPACE}/sbt/sbt -DsparkVersion=${v} 'project spark_22' 'sonatypeRelease'"
        }
    } 
}

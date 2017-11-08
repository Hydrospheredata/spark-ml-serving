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

projects = [
   "common",
   "spark_20",
   "spark_21",
   "spark_22"
]

node("JenkinsOnDemand") {
    def repository = 'spark-ml-serving'
    def organization = 'Hydrospheredata'
    def gitCredentialId = 'HydrospheredataGithubAccessKey'

    def sbtOpts = "-Dsbt.override.build.repos=true -Dsbt.repository.config=${env.WORKSPACE}/project/repositories"
    stage('Checkout') {
        checkoutSource(gitCredentialId, organization, repository)
    }

    stage('Test') {
        for (int i = 0; i < projects.size(); i++) { //TODO switch to each after JENKINS-26481
            def project = projects.get(i)         
            sh "${env.WORKSPACE}/sbt/sbt ${sbtOpts} ${project}/test"
        }
    }

    if (isReleaseJob()) {
        stage("Publish"){
            for (int i = 0; i < projects.size(); i++) { //TODO switch to each after JENKINS-26481
                def project = projects.get(i)         
                sh "${env.WORKSPACE}/sbt/sbt ${sbtOpts} 'set pgpPassphrase := Some(Array())' ${project}/publishSigned"
                sh "${env.WORKSPACE}/sbt/sbt ${sbtOpts} 'project ${project}' 'sonatypeRelease'"
            }
        }
    } 
}

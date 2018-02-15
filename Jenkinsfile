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
    def tag = sh(returnStdout: true, script: "git tag -l --contains HEAD").trim()
    return tag.startsWith("v")
}

sparkVersions = [
        "2.0.2",
        "2.1.2",
        "2.2.0"
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
        for (int i = 0; i < sparkVersions.size(); i++) { //TODO switch to each after JENKINS-26481
            def sparkV = sparkVersions.get(i)
            sh "${env.WORKSPACE}/sbt/sbt ${sbtOpts} -DsparkVersion=${sparkV} test"
        }
    }

    if (isReleaseJob()) {
        stage("Publish"){
            for (int i = 0; i < sparkVersions.size(); i++) { //TODO switch to each after JENKINS-26481
                def sparkV = sparkVersions.get(i)
                sh "${env.WORKSPACE}/sbt/sbt ${sbtOpts} -DsparkVersion=${sparkV} 'set pgpPassphrase := Some(Array())' publishSigned"
                sh "${env.WORKSPACE}/sbt/sbt ${sbtOpts} -DsparkVersion=${sparkV} 'sonatypeRelease'"
            }
        }
    } 
}

This section provides details on how to configure and/or develop this codebase.

## Continuos Integration/Continuous Deployment (CI/CD) Workflow

This project uses *GitHub Workflows* to automate or ease a number of development tasks.  These
workflows can be found within the `./.github/workflows/` directory and include:
1. **pull_request.yml**

    This workflow is run whenever a pull request is opened, updated or reopened.  The following things are enforced by this workflow:

    - linting and code formatting standards
    - proper maintainance of the Poetry project
    - successful building of the project
    - successful building of the documentation
    - successful running of tests

2. **bump.yml**

    This workflow leverages the colocated `bump.sh` bash script to automatically increment the project version whenever code is pushed to the `main` branch.  It is controlled by adding the text `[version:minor]` or `[version:major]` to one of the commit messages of a pull request.

3. **publish.yml**

    This workflow is run whenever a new release is generated through *GitHub* (see below for details on how to do this).  Documentation is updated on *Read the Docs* and a new version of the code is published on the *Python Package Index* (*PyPI*).

## Setting-up the Code

A local development copy of the code base can be obtained and configured as follows:

* Navigate to the *GitHub* page hosting the project
* If you want to fork the code so that you work on your own version of the repository (not generally needed or
  recommended):
    - Click on the `fork` button at the top of the page;
    - Edit the details you want to have for the new repoitory; and
    - Press `Create fork`.
* Obtain the URL for the repository you're going to use (denoted `<url>`) by clicking on the green `Code` button on the repository *GitHub* page
* On your local machine, navigate with your terminal to the location where you want to place the code
* Generate a local copy using `git clone <url>`;

::: {note}
Although not strictly necessary, it is recommended that you configure the branch permissions of any forked repositories as detailed in the *GitHub* configuration section below.
:::

## Poetry and Python environments for development

Poetry is used to manage this project ([see here for an introduction](https://python-poetry.org)).  It simplifies & helps with managing the following:

1. **Creation and activation of a Python environment for the project**

    Python development should always be managed using a Python environment.  Poetry makes this easy for you.  You simply run the following from within the project:

    ``` console
    $ poetry shell
    ```
    
    ::: {note}
    You don't have to use Poetry to manage your Python environment if you would rather not.  You can instruct Poetry to respect your Python environemnts (e.g. created with `pyenv`) by setting the following option:
    ``` console
    $ poetry config virtualenvs.prefer-active-python true
    ```
    :::

2. **Dependency management**

    Poetry manages a "lock file" (which should be committed and maintained within the code repository) ensuring repeatible installs for all versions.

3. **Publication of the project to the Python Package Index (*PyPI*) so that people can easily install it for themselves**

    Once properly configured, publishing to *PyPI* with Poetry is extremely easy.  This is generally managed by the CI/CD workflow for the project though, and developers should never have to manually do this.

## Installing Development Dependencies

Once the code is locally installed, development dependencies should be installed by moving to the project's root directory and executing the following:

``` console
$ poetry install --all-extras
```

In what follows, it will be assumed that this has been done.

## Guidelines

In the following, we lay-out some important guidelines for developing on this codebase.

### Branches

***Development should never be conducted on the `main` branch***.  If *GitHub* has been properly configured (see below), then merges to this branch are limited to Pull Requests (PRs) only.  Once a PR is opened for the `main` branch, the project tests are run.  When it is closed and code is committed to the main branch, the project version is automatically incremented (see below).

### Versioning

Semantic versioning (i.e. a scheme that follows a `vMAJOR.MINOR.PATCH` format; see <https://semver.org> for details) is used for this project.  ***The single point of truth for the current production version is the last git tag on the main branch with a `v[0-9]*` format***.  When developing locally, the reported version will often appear as `v0.0.0-dev`.

Changes are handled by a *GitHub Workflow* which increments the version and creates a new tag whenever a push occurs to the `main` branch.  This ensures that every commit on the `main` branch is assigned a unique version.  The logic by which it modifies the version is as follows:

1. if the PR message (or one of its commits' messages) contains the text `[version:major]`, then `MAJOR` is incremented;
2. else if the PR message (or one of its commits' messages) contains the text `[version:minor]`, then `MINOR` is incremented;
3. else `PATCH` is incremented.

A `MAJOR` version change should be indicated if the PR introduces a breaking change.  A `MINOR` version change should be indicated if the PR introduces new functionality.

::: {note}
Make sure you think carefully about what type of changes you are committing.  If you are adding functionality, make sure you bump the **MINOR** version; if you are making breaking changes, make sure you bump the **MAJOR** version.  ***Users will be very thankful that you did!***
:::

### Tests

*PyTest* is used to run tests for this codebase.  Make sure you run them before submitting any code to a PR by
executing the following from the project root directory:

``` console
$ pytest
```

Some further comments about how testing has been configured for this project:

#### Coverage

*PyTest* has been configured for this project to create a coverage report after running.  This report will inform the developer of what fraction of the code base is exercised by the tests and give a list of lines of code in each Python filename which has not been exercised by the tests run.

While not strictly enforced, we encourage developers to make sure that anything they do to the codebase does not reduce this metric.  This report can be used to inform what parts of the codebase need further testing.

### Git Hooks

This project has been set-up with pre-configured git hooks. They should be used as a means for developers to quickly check that (at least some) of the code standards of the project are being met by commited code.  Ultimately, all standards are actually enforced by the continuous integration pipeline (see below).  Running quick checks (like linting) at the point of commiting code can save time that might otherwise be lost later (for example) at the PR or release stage when testing needs to be rigorous and policy enforcement generally fails slower.  Developers can choose to either:

1. use the git hooks defined by this project (recommended, for the reasons given above; see below for instructions),
2. not to use them, and rely purely on the CI workflow to enforce all project policies, or
3. configure their IDE of choice to manage things, in which case it is up to them to make sure that this aligns with the policies being enforced by the CI.

If developers would like to utilise the git hooks provided by this project they just need to run the following command from within the project:
``` console
$ pre-commit install
```

Some of these hooks require internet access to work.  If you are trying to commit to the
repository locally and are being prevented from doing so because you are working online, the
hooks can be ignored by using the `--no-verify` flag when running `git commit`, like so:
``` console
$ git commit --no-verify
```

Alternatively, you can disable them by running:
``` console
$ pre-commit uninstall
```

They can subsequently be re-enabled by reinstalling them.

#### Maintaining Git Hooks

The git hooks are defined in the `.pre-commit-config.yaml` file.  Specific revisions for many of the tools listed should be managed with Poetry, with syncing managed with the [sync_with_poetry](https://github.com/floatingpurr/sync_with_poetry) hook.  Developers should take care not to use git hooks to *enforce* any project policies.  That should all be done within the continuous integration workflows.  Instead: these should just be quality-of-life checks that fix minor issues or prevent the propagation of quick-and-easy-to-detect problems which would otherwise be caught by the CI later with considerably more latency.  Furthermore, ensure that the checks performed here are consistant between the hooks and the CI.  For example: make sure that any linting/code quality checks are executed with the same tools and options.

### Releases

Releases are generated through the *GitHub* UI.  A *GitHub Workflow* has been configured to do the following when a new release is produced:

1. Run the tests for the project,
2. Ensure that the project builds,
3. Rebuild the documentation on *Read the Docs*, and
4. Publish a new version of the code on [*PyPI*](https://pypi.org/).

::: {note}
If a release is flagged as a "pre-release" through the *GitHub* interface, then documentation will not be built and the project will be published on *test.PyPI.org* instead.
:::

#### Generating a new release

To generate a new release, do the following:

1. Navigate to the project's *GitHub* page,
2. Click on `Releases` in the sidebar,
3. Click on `Create a new release` (if this is the first release you have generated) or `Draft release` if this is a subsequent release,
4. Click on `Choose a tag` and select the most recent version listed,
5. Write some text describing the nature of the release to prospective users, and
6. Click `Publish Release`.

### Documentation

Documentation for this project is generated using [Sphinx](https://www.sphinx-doc.org/en/master/) and is hosted on *Read the Docs* for the latest release version.  Sphinx is configured here using [*MyST-Parser*](https://myst-parser.readthedocs.io/en/latest/)so that content can be managed with Markdown (`.md`) rather than Restructured Text (`.rst`).  *MyST-Parser* also offers [several optional Markdown extensions](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html) enabling the rendering of richer content (e.g. Latex equations).  Several of these extensions have been enabled by default, but not all.  This can be managed by overriding the behavior of the `conf.py` template (see below for directions on overriding templates) and editing the `myst_enable_extensions` list therein.

#### Generating the Documentation

Documentation can be generated locally by running the following from the `docs` directory of the project:
``` console
$ make html
```

This will generate an html version of the documentation in `docs/_build/html` which can be opened in your browser.  On a Mac (for example), this can be done by running the following:
``` console
$ open docs/_build/html/index.html
```

## Configuring Services

Develpers and project owners/maintainers will require accounts with one or all of the following services to work with this codebase.  This section details how these services need to be configured.  Following these steps should only be necessarry - or partially necessary - if a developer chooses to fork the project.

1. [***GitHub***](https:/github.com)

    To work with this codebase, you will require a *GitHub* account ([go here to get one](https://github.com)).
    
    Branch permissions for the project repository should be configured as follows:
    
    - Protect the main branch to only permit merges from pull requests.  This can be done by clicking on the 'branches' tab and clicking on the 'Protect this branch' button for the 'main' branch.
    - Select 'Require status checks to pass before merging' when you set-up this branch protection rule.  This will ensure that all CI/CD tests pass before a merge to the main branch can be made.
    
    Several secrets need to be configured by navigating to `Settings->Secrets->Actions` and adding the following:
    
    - To host the project documentation on *Read the Docs** (see below), the following secrets need to be set (see below for where to find these values):
    
        - **RTD_WEBHOOK_TOKEN**, and
        - **RTD_WEBHOOK_URL**
    
    - To make code releases available on the **Python Package Index** (see below), then the following secret needs to be set (see below for where to find this value):
    
        - **PYPI_TOKEN**,
    
    - To test code releases with the **Test Python Package Index** (see below), then the following secret needs to be set (see below for where to find this value):
    
        - **TEST_PYPI_TOKEN**,

2. [__Read the Docs__](https://readthedocs.org)

    **Read the Docs** (*RTD*) is used to build and host the project documentation.  An account is needed if you are an owner/maintainer of the project and will be publishing and managing the project's documentation online, but not needed if you are simply a contributing developer.  *RTD* can be configured in either of the following ways:

    1. **By connecting *RTD* to your *GitHub* account**
        - Ensure that your *GitHub* account has been connected.  This is done automatically if you log into *RTD* with your *GitHub* credentials.  If you logged in with your email, navigate to `<login_id>->Settings->Connected Services` by clicking on "Connect Your Accounts" and click "Connect to GitHub".  You know your account is linked if it is listed below under "Active Services".

        - Return to your *RTD* landing page by clicking on your account name at the top.  Click "Import a Project".  Your *GitHub* repository should be listed here (you may need to refresh the list if it has been created recently).  Import it.

        - To obtain **RTD_WEBHOOK_TOKEN**, navigate to `<Account>->Settings->API Tokens` on *Read the Docs*.  If a token has been created already, you can use it.  Otherwise (or if you want a token specifically for this project), create a new one.

        - To obtain **RTD_WEBHOOK_URL**, migrate to the `Admin->Integrations` tab on the *RTD* project page.  Click on your incomming webhook and get the URL there.

    2. **By creating a Generic Webhook**
        - Navigate to the `Admin->Integrations` tab on the *RTD* project page and click `Add integration`.  Then select `Generic API incoming webhook` from the dropdown and click `Add integration`.

        - To obtain **RTD_WEBHOOK_URL** and **RTD_WEBHOOK_TOKEN**, migrate to the `Admin->Integrations` tab on the *RTD* project page and click on your incomming webhook.

    Once properly configured, the documentation for this project should build automatically on *RTD* every time you generate a new release (see below for instructions).

    ::: {note}
    Make sure **RTD_WEBHOOK_URL** starts with `https://`.  Prepend it if not.
    :::

3. The [__Python Package Index (*PyPI*)__](https://pypi.org)

    This service is used to publish project releases.  An account is needed if you are the owner of the project, but not generally needed if you are simply a contributing developer.  An API token will need to be created and added to your *GitHub* project as **PYPI_TOKEN** (as detailed above).  This can be generated from the *PyPI* UI by navigating to `Account Settings->Add API Token`.

    To test releases, a parallel account on *test.PyPI* is needed and a similar token to **PYPI_TOKEN** - named **TEST_PYPI_TOKEN** needs to be set, in the same way as above.  To create a test release, flag it as a "pre-release" through the *GitHub* interface when you generate a release, and it will be published on *test.PyPI.org* rather than *PyPI.org*.

    ::: {note}
    Although `poetry` can be used to directly publish this project to *PyPI*, users should not do this.  The proper way to publish the project is through the *GitHub* interface, which leverages the *GitHub Workflows* of this project to ensure the enforcement of project standards before a new version can be created.
    :::

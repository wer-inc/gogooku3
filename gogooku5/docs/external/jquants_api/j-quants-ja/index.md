<!DOCTYPE html><html lang="en" class="notranslate" translate="no"><head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GitBook</title>
    <link rel="manifest" href="/public/manifest.json">
    <link rel="icon" sizes="512x512" href="/public/images/icon-512.png" media="(prefers-color-scheme: light)">
    <link rel="icon" sizes="512x512" href="/public/images/icon-512-dark.png" media="(prefers-color-scheme: dark)">
    <link rel="apple-touch-icon" sizes="512x512" href="/public/images/icon-ios/icon_512x512.png">
    <link rel="apple-touch-icon" sizes="512x512@2x" href="/public/images/icon-ios/icon_512x512@2x.png">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="GitBook">
    <meta name="theme-color" content="#f7f7f7">
    <meta name="description" content="GitBook">
    <link rel="preconnect" href="https://api.gitbook.com">
    <link rel="preconnect" href="https://content.gitbook.com">
    <script type="text/javascript" defer="" src="https://cdn.iframe.ly/embed.js" async=""></script>
    <!--
      Google Tag Manager tracking script to track conversions from the site.
      See https://gitbook.slack.com/archives/C07AQA4256G/p1721923712258389 for more info
    -->
    <script>
      (function (w, d, s, l, i) {
          w[l] = w[l] || [];
          w[l].push({ 'gtm.start': new Date().getTime(), event: 'gtm.js' });
          var f = d.getElementsByTagName(s)[0],
              j = d.createElement(s),
              dl = l != 'dataLayer' ? '&l=' + l : '';
          j.async = true;
          j.src = 'https://www.googletagmanager.com/gtm.js?id=' + i + dl;
          f.parentNode.insertBefore(j, f);
      })(window, document, 'script', 'dataLayer', 'GTM-PVD2ZHVC');
  </script>
    <script>
          (async function() {
          // Splash screen modifications

          // 1. Adapt to dark/light theme
          const theme = localStorage.getItem('@recoil/userThemeAtom');
          if (theme?.includes('dark')) {
            document.documentElement.classList.add('theme-color-dark');
          } else if (theme?.includes('light')) {
            document.documentElement.classList.add('theme-color-light');
          }

          function hideSidebar() {
              const sidebar = document.querySelector('.sidebar');
              if (sidebar) sidebar.style.display = 'none';
          }

          function applySidebarSizing() {
              let sidebarWidth;
              let isSidebarCollapsed = false;

              function applySidebarWidth() {
                  const sidebar = document.querySelector('.sidebar');
                  if (!sidebar) return;

                  if (isSidebarCollapsed) {
                      sidebar.style.setProperty('--sidebar-width', '0px')
                  } else if (sidebarWidth) {
                      sidebar.style.setProperty('--sidebar-width', sidebarWidth + 'px');
                  }
              }

              try {
                  const dbName = 'keyval-store';
                  const storeName = 'keyval';
                  const request = indexedDB.open(dbName, 1);

                  request.onupgradeneeded = (event) => {
                      const db = event.target.result;
                      if (!db.objectStoreNames.contains(storeName)) {
                          db.createObjectStore(storeName);
                      }
                  };

                  request.onsuccess = (event) => {
                      const db = event.target.result;
                      if (db.objectStoreNames.contains(storeName)) {
                          const transaction = db.transaction(storeName, 'readonly');
                          const store = transaction.objectStore(storeName);

                          const widthRequest = store.get('@recoil/sidebarWidth');
                          widthRequest.onsuccess = () => {
                              if (widthRequest.result !== undefined) {
                                  sidebarWidth = widthRequest.result;
                              }
                              const collapsedRequest = store.get('@recoil/sidebarCollapsed');
                              collapsedRequest.onsuccess = () => {
                                  if (collapsedRequest.result !== undefined) {
                                      isSidebarCollapsed = collapsedRequest.result;
                                  }
                                  applySidebarWidth();
                              };
                          };
                      } else {
                          applySidebarWidth();
                      }
                  };

                  request.onerror = () => {
                      applySidebarWidth();
                  };

              } catch (e) {
                  applySidebarWidth();
              }
          }

          // Leave early if no indexedDB
          if (!('indexedDB' in window)) {
              hideSidebar();
              return;
          }

        const path = window.location.pathname;
        const urlParams = new URLSearchParams(window.location.search);
        const hasAuthToken = urlParams.has('auth_token');

        // Detect SAML auth flows (auth_token without apiTestMode) to skip problematic Firebase checks
        const isSamlAuth = hasAuthToken && !urlParams.has('apiTestMode');

        // 2. Check auth state. Sign-in page layout doesn't match editor, so we need to hide the sidebar there. Skip Firebase auth checks for SAML flows to avoid race conditions in CI
        if (!isSamlAuth) {
            try {
                const firebaseDbName = 'firebaseLocalStorageDb';
                const firebaseRequest = indexedDB.open(firebaseDbName, 1);

                // Add timeout to prevent blocking page render in CI
                const authCheckTimeout = setTimeout(() => {
                    hideSidebar();
                }, 1000);

                firebaseRequest.onsuccess = (event) => {
                    clearTimeout(authCheckTimeout);
                    const db = event.target.result;
                    const objectStoreNames = Array.from(db.objectStoreNames);

                    if (!objectStoreNames.length) {
                        return hideSidebar();
                    }

                    const transaction = db.transaction(objectStoreNames, 'readonly');
                    const store = transaction.objectStore(objectStoreNames[0]);

                    // Look for Firebase auth user key pattern
                    const getAllRequest = store.getAllKeys();
                    getAllRequest.onsuccess = () => {
                    const keys = getAllRequest.result;
                    const authKey = keys.find(key =>
                        typeof key === 'string' &&
                        key.includes('firebase:authUser:') &&
                        key.includes('[DEFAULT]')
                    );

                    if (!authKey) {
                        return hideSidebar();
                    }

                    // Check if the auth user actually has data
                    const getRequest = store.get(authKey);
                    getRequest.onsuccess = () => {
                        if (!getRequest.result) {
                            return hideSidebar();
                        }
                    // At this point, user is logged in. Apply sidebar width/collapsed logic.
                    applySidebarSizing();
                    };
                    };
                };

                firebaseRequest.onerror = () => {
                    clearTimeout(authCheckTimeout);
                    hideSidebar();
                };
            } catch (e) {
                clearTimeout(authCheckTimeout);
                hideSidebar();
            }
        } else {
            // For SAML auth flows, still apply sidebar sizing but skip Firebase auth checks
            // This ensures SAML users get their sidebar preferences without CI race conditions
            applySidebarSizing();
        }

        // Adds header nav bars (Space, OpenAPI, Site, Integration detail page)
        const pagesWithHeader = ['/openapi/', '/s/', '/site', '/integrations/'];
        const showSpaceheader = pagesWithHeader.some(route => path.includes(route));

        if (showSpaceheader) {
            document.documentElement.classList.add('show-spaceheader');
        }
    })();
    </script>
  <link rel="stylesheet" href="/public/dist/index-HPKA2IXK.css"></head>
  <body>
    <!-- Google Tag Manager -->
    <noscript>
      <iframe src="https://www.googletagmanager.com/ns.html?id=GTM-PVD2ZHVC" height="0" width="0" style="display:none;visibility:hidden"></iframe>
    </noscript>

    <div id="gitbook-root"></div>
    <div class="gitbook-splashscreen">
      <div class="sidebar">
        <div class="sidebar-header-skeleton">
            <div class="org-switcher-skeleton shimmer-dark"></div>
        </div>
      </div>
      <div class="application">
        <div class="spaceheader"></div>
      </div>
    </div>




<script src="/public/dist/index-52KN2HGX.min.js" type="module"></script></body></html>

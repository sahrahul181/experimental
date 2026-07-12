$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Building ART dex file from libcore sources " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Locate Android SDK tools
$sdkPath = "C:\Users\RS\AppData\Local\Android\Sdk"
if (-not (Test-Path $sdkPath)) {
    Write-Error "Android SDK not found at $sdkPath"
}

# Auto-detect latest core-for-system-modules.jar (contains internal Android & ICU class definitions)
$systemJars = Get-ChildItem -Path "$sdkPath\platforms" -Filter "core-for-system-modules.jar" -Recurse -File
if ($systemJars.Count -eq 0) {
    Write-Error "Could not find core-for-system-modules.jar in $sdkPath\platforms"
}
$systemJar = $systemJars | Sort-Object -Property FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
Write-Host "Using system jar dependency: $systemJar" -ForegroundColor Green

# Auto-detect latest d8.bat
$d8s = Get-ChildItem -Path "$sdkPath\build-tools" -Filter "d8.bat" -Recurse -File
if ($d8s.Count -eq 0) {
    Write-Error "Could not find d8.bat in $sdkPath\build-tools"
}
$d8 = $d8s | Sort-Object -Property FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
Write-Host "Using d8: $d8" -ForegroundColor Green

# 2. Create stub directories and annotation/class files
$stubsDir = "stubs"
Write-Host "Creating stub classes for internal annotations, flags, and missing APIs..." -ForegroundColor Cyan
if (Test-Path $stubsDir) { Remove-Item -Recurse -Force $stubsDir }

# Define all stub directories
$stubSubdirs = @(
    "android\annotation",
    "android\compat\annotation",
    "com\android\libcore",
    "com\android\art\flags",
    "com\android\okhttp\internalandroidapi",
    "com\android\i18n\system",
    "com\android\i18n\timezone",
    "com\android\icu\util",
    "com\android\icu\util\regex",
    "com\android\icu\text",
    "com\android\icu\charset",
    "java\lang",
    "libcore\icu"
)

foreach ($dir in $stubSubdirs) {
    New-Item -ItemType Directory -Force -Path "$stubsDir\$dir" | Out-Null
}

# Stub for FlaggedApi
Set-Content -Path "$stubsDir\android\annotation\FlaggedApi.java" -Value @"
package android.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface FlaggedApi {
    String value();
}
"@

# Stub for TestApi
Set-Content -Path "$stubsDir\android\annotation\TestApi.java" -Value @"
package android.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface TestApi {}
"@

# Stub for UserIdInt
Set-Content -Path "$stubsDir\android\annotation\UserIdInt.java" -Value @"
package android.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface UserIdInt {}
"@

# Stub for generated Feature Flags
Set-Content -Path "$stubsDir\com\android\libcore\Flags.java" -Value @"
package com.android.libcore;
public class Flags {
    public static final String FLAG_MADVISE_API = "com.android.libcore.madvise_api";
    public static final String FLAG_HPKE_PUBLIC_API = "com.android.libcore.hpke_public_api";
    public static final String FLAG_NATIVE_METRICS = "com.android.libcore.native_metrics";
    public static final String FLAG_POST_CLEANUP_APIS = "com.android.libcore.post_cleanup_apis";
    public static final String FLAG_OPENJDK_21_V1_APIS = "com.android.libcore.openjdk_21_v1_apis";
    public static boolean scheduleAtFixedRateNewBehavior() { return false; }
    public static boolean vApis() { return false; }
    public static boolean readOnlyDynamicCodeLoad() { return false; }
}
"@

# Stub for ART flags
Set-Content -Path "$stubsDir\com\android\art\flags\Flags.java" -Value @"
package com.android.art.flags;
public class Flags {
    public static boolean test() { return false; }
}
"@

# Stub for OkHttp Dns
Set-Content -Path "$stubsDir\com\android\okhttp\internalandroidapi\Dns.java" -Value @"
package com.android.okhttp.internalandroidapi;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.List;
public interface Dns {
    List<InetAddress> lookup(String hostname) throws UnknownHostException;
}
"@

# Stub for OkHttp HTTP connection factory
Set-Content -Path "$stubsDir\com\android\okhttp\internalandroidapi\HttpURLConnectionFactory.java" -Value @"
package com.android.okhttp.internalandroidapi;
import java.net.*;
import javax.net.SocketFactory;
import javax.net.ssl.*;
import java.util.concurrent.TimeUnit;
import libcore.net.http.Dns;
public class HttpURLConnectionFactory {
    public HttpURLConnectionFactory() {}
    public void setNewConnectionPool(int maxIdleConnections, long keepAliveDuration, TimeUnit timeUnit) {}
    public void setDns(Dns dns) {}
    public HttpURLConnection openConnection(URL url, SocketFactory socketFactory, Proxy proxy) throws java.io.IOException { return null; }
}
"@

# Stubs for i18n hooks
Set-Content -Path "$stubsDir\com\android\i18n\system\AppSpecializationHooks.java" -Value @"
package com.android.i18n.system;
public class AppSpecializationHooks {
    public static void handleCompatChangesBeforeBindingApplication() {}
}
"@

Set-Content -Path "$stubsDir\com\android\i18n\system\ZygoteHooks.java" -Value @"
package com.android.i18n.system;
public class ZygoteHooks {
    public static void onBeginPreload() {}
    public static void onEndPreload() {}
}
"@

# Stub for internal system annotations
Set-Content -Path "$stubsDir\android\annotation\SystemApi.java" -Value @"
package android.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface SystemApi {
    Client client() default Client.PRIVILEGED_APPS;
    enum Client {
        PRIVILEGED_APPS, MODULE_LIBRARIES, SYSTEM_SERVER
    }
}
"@

Set-Content -Path "$stubsDir\android\compat\annotation\UnsupportedAppUsage.java" -Value @"
package android.compat.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface UnsupportedAppUsage {
    long trackingBug() default 0;
    int implicitMemberSignatureRules() default 0;
    String publicAlternative() default "";
    String publicAlternatives() default "";
    int maxTargetSdk() default 0;
}
"@

Set-Content -Path "$stubsDir\android\compat\annotation\ChangeId.java" -Value @"
package android.compat.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface ChangeId {}
"@

Set-Content -Path "$stubsDir\android\compat\annotation\EnabledSince.java" -Value @"
package android.compat.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface EnabledSince {
    int targetSdkVersion();
}
"@

Set-Content -Path "$stubsDir\android\compat\annotation\EnabledAfter.java" -Value @"
package android.compat.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface EnabledAfter {
    int targetSdkVersion();
}
"@

Set-Content -Path "$stubsDir\android\compat\annotation\Disabled.java" -Value @"
package android.compat.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.CLASS)
public @interface Disabled {}
"@

# ICU and I18n Native Stubs
Set-Content -Path "$stubsDir\com\android\icu\util\ExtendedCalendar.java" -Value @"
package com.android.icu.util;
import android.icu.util.ULocale;
public class ExtendedCalendar {
    public static ExtendedCalendar getInstance(ULocale uLocale) { return null; }
    public String getDateTimePattern(int dateStyle, int timeStyle) { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\text\ExtendedDateFormatSymbols.java" -Value @"
package com.android.icu.text;
import android.icu.util.ULocale;
import android.icu.text.DateFormatSymbols;
public class ExtendedDateFormatSymbols {
    public static ExtendedDateFormatSymbols getInstance(ULocale uLocale) { return null; }
    public DateFormatSymbols getDateFormatSymbols() { return null; }
    public String[] getNarrowQuarters(int context) { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\text\ExtendedDecimalFormatSymbols.java" -Value @"
package com.android.icu.text;
import android.icu.util.ULocale;
import android.icu.text.NumberingSystem;
public class ExtendedDecimalFormatSymbols {
    public static ExtendedDecimalFormatSymbols getInstance(ULocale uLocale, NumberingSystem ns) { return null; }
    public String getLocalizedPatternSeparator() { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\util\LocaleNative.java" -Value @"
package com.android.icu.util;
import java.util.Locale;
public class LocaleNative {
    public static void setDefault(String languageTag) {}
    public static String getDisplayLanguage(Locale l1, Locale l2) { return null; }
    public static String getDisplayScript(Locale l1, Locale l2) { return null; }
    public static String getDisplayCountry(Locale l1, Locale l2) { return null; }
    public static String getDisplayVariant(Locale l1, Locale l2) { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\util\regex\PatternNative.java" -Value @"
package com.android.icu.util.regex;
public class PatternNative {
    public static PatternNative create(String icuPattern, int icuFlags) { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\util\regex\MatcherNative.java" -Value @"
package com.android.icu.util.regex;
public class MatcherNative {
    public static MatcherNative create(PatternNative pat) { return null; }
    public int groupCount() { return 0; }
    public boolean matches(int[] groups) { return false; }
    public boolean findNext(int[] groups) { return false; }
    public boolean find(int start, int[] groups) { return false; }
    public boolean lookingAt(int[] groups) { return false; }
    public int getMatchedGroupIndex(String name) { return 0; }
    public void useTransparentBounds(boolean b) {}
    public void useAnchoringBounds(boolean b) {}
    public boolean hitEnd() { return false; }
    public boolean requireEnd() { return false; }
    public void setInput(String text, int from, int to) {}
}
"@

Set-Content -Path "$stubsDir\com\android\icu\util\ExtendedTimeZone.java" -Value @"
package com.android.icu.util;
public class ExtendedTimeZone {
    public static ExtendedTimeZone getInstance(String zoneId) { return null; }
    public java.time.zone.ZoneRules createZoneRules() { return null; }
    public static void clearDefaultTimeZone() {}
}
"@

Set-Content -Path "$stubsDir\com\android\icu\charset\CharsetFactory.java" -Value @"
package com.android.icu.charset;
import java.nio.charset.Charset;
public class CharsetFactory {
    public static Charset create(String charsetName) { return null; }
    public static String[] getAvailableCharsetNames() { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\text\ExtendedIDNA.java" -Value @"
package com.android.icu.text;
import android.icu.text.StringPrepParseException;
public class ExtendedIDNA {
    public static StringBuffer convertIDNToASCII(CharSequence input, int flag) throws StringPrepParseException { return null; }
    public static StringBuffer convertIDNToUnicode(CharSequence input, int flag) throws StringPrepParseException { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\text\CompatibleDecimalFormatFactory.java" -Value @"
package com.android.icu.text;
public class CompatibleDecimalFormatFactory {
    public static android.icu.text.DecimalFormat create(String pattern, android.icu.text.DecimalFormatSymbols dfs) { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\text\ExtendedTimeZoneNames.java" -Value @"
package com.android.icu.text;
import android.icu.util.ULocale;
public class ExtendedTimeZoneNames {
    public static ExtendedTimeZoneNames getInstance(ULocale uLocale) { return null; }
    public Match matchName(String text, int start, String tzId) { return null; }
    public android.icu.text.TimeZoneNames getTimeZoneNames() { return null; }
    public static class Match {
        public Match() {}
        public String getTzId() { return null; }
        public boolean isDst() { return false; }
        public int getMatchLength() { return 0; }
    }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\text\TimeZoneNamesNative.java" -Value @"
package com.android.icu.text;
import java.util.Locale;
public class TimeZoneNamesNative {
    public static String[][] getFilledZoneStrings(Locale locale, String[] ids) { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\icu\util\CaseMapperNative.java" -Value @"
package com.android.icu.util;
import java.util.Locale;
public class CaseMapperNative {
    public static String toLowerCase(String s, Locale locale) { return null; }
    public static String toUpperCase(String s, Locale locale) { return null; }
}
"@

Set-Content -Path "$stubsDir\com\android\i18n\timezone\ZoneInfoData.java" -Value @"
package com.android.i18n.timezone;
import java.io.IOException;
import java.io.ObjectOutputStream;
public class ZoneInfoData {
    public static final String[] ZONEINFO_SERIALIZED_FIELDS = new String[0];
    public static ZoneInfoData createFromSerializationFields(String id, Object fields) { return null; }
    public Integer getLatestDstSavingsMillis(long timeInMillis) { return null; }
    public long[] getTransitions() { return null; }
    public String getID() { return null; }
    public void writeToSerializationFields(ObjectOutputStream.PutField putField) throws IOException {}
    public int getRawOffset() { return 0; }
    public int getOffset(long time) { return 0; }
    public boolean isInDaylightTime(long time) { return false; }
    public ZoneInfoData createCopyWithRawOffset(int off) { return null; }
    public boolean hasSameRules(ZoneInfoData other) { return false; }
    public int getOffsetsByUtcTime(long utcTimeInMillis, int[] offsets) { return 0; }
}
"@

Set-Content -Path "$stubsDir\com\android\i18n\timezone\ZoneInfoDb.java" -Value @"
package com.android.i18n.timezone;
public class ZoneInfoDb {
    public static ZoneInfoDb getInstance() { return null; }
    public ZoneInfoData makeZoneInfoData(String id) { return null; }
    public String[] getAvailableIDs(int rawOffset) { return null; }
    public String[] getAvailableIDs() { return null; }
}
"@

Set-Content -Path "$stubsDir\libcore\icu\TimeZoneNamesNative.java" -Value @"
package libcore.icu;
import java.util.Locale;
public class TimeZoneNamesNative {
    public static String[][] getFilledZoneStrings(Locale locale, String[] ids) { return null; }
}
"@

# 3. Find and list all Java source files from our git checkout
Write-Host "Gathering Java source files..." -ForegroundColor Cyan
$sourceDirs = @(
    "libcore/ojluni/src/main/java",
    "libcore/luni/src/main/java",
    "libcore/dalvik/src/main/java",
    "libcore/libart/src/main/java",
    "libcore/json/src/main/java",
    "libcore/xml/src/main/java",
    "stubs"
)

$javaFiles = Get-ChildItem -Path $sourceDirs -Filter *.java -Recurse -File | Resolve-Path -Relative
$javaFiles | Out-File -FilePath sources.txt -Encoding ascii
Write-Host "Found $($javaFiles.Count) Java source files to compile." -ForegroundColor Green

# 4. Compile Java files
Write-Host "Compiling Java files..." -ForegroundColor Cyan
$classesDir = "classes"
if (Test-Path $classesDir) { Remove-Item -Recurse -Force $classesDir }
New-Item -ItemType Directory -Path $classesDir | Out-Null

# We patch java.base with all our local sources.
# The system jar is put on the classpath to resolve com.android.icu.* etc.
$patchPath = $sourceDirs -join ";"

# Run javac
$javacArgs = @(
    "-d", $classesDir,
    "-classpath", $systemJar,
    "--patch-module", "java.base=$patchPath",
    "--add-reads", "java.base=ALL-UNNAMED",
    "--add-exports", "java.base/jdk.internal.vm.annotation=ALL-UNNAMED",
    "@sources.txt"
)

$process = Start-Process -FilePath "javac" -ArgumentList $javacArgs -PassThru -NoNewWindow -Wait
if ($process.ExitCode -ne 0) {
    Write-Error "Java compilation failed with exit code $($process.ExitCode)"
}
Write-Host "Java files compiled successfully." -ForegroundColor Green

# 5. Package into classes.dex
Write-Host "Packaging classes into DEX format..." -ForegroundColor Cyan
if (Test-Path "classes.dex") { Remove-Item -Force "classes.dex" }

# Zip the compiled classes to prevent command line length limitations and d8 directory import issues
$tempZip = "temp_classes.zip"
if (Test-Path $tempZip) { Remove-Item -Force $tempZip }
Compress-Archive -Path "$classesDir\*" -DestinationPath $tempZip

# Run d8
$d8Args = @(
    "--output", ".",
    "--min-api", "26",
    $tempZip
)

$process = Start-Process -FilePath $d8 -ArgumentList $d8Args -PassThru -NoNewWindow -Wait
if ($process.ExitCode -ne 0) {
    Remove-Item -Force $tempZip
    Write-Error "DEX packaging (d8) failed with exit code $($process.ExitCode)"
}

Remove-Item -Force $tempZip

if (-not (Test-Path "classes.dex")) {
    Write-Error "classes.dex was not generated!"
}
Write-Host "classes.dex generated successfully!" -ForegroundColor Green
Copy-Item -Path "classes.dex" -Destination "core.dex" -Force
Write-Host "core.dex updated successfully!" -ForegroundColor Green

# 6. Cleanup temporary files
Write-Host "Cleaning up build files..." -ForegroundColor Cyan
Remove-Item -Recurse -Force $stubsDir
Remove-Item -Recurse -Force $classesDir
Remove-Item -Force sources.txt

Write-Host "=============================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESSFUL: classes.dex is ready!    " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

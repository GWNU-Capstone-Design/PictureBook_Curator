<%@ page import="java.util.List" %>
<%@ page import="java.util.ArrayList" %>
<%@ page import="java.io.IOException" %>
<%@ page import="java.sql.*" %>
<%@ page import="Database.DatabaseConnector" %>
<%@ page import="com.google.gson.Gson" %>
<%@ page contentType="application/json; charset=UTF-8" %>
<%
    // 세션에서 사용자 아이디 가져오기
    Integer userId = (Integer) request.getSession().getAttribute("user_id");
    // 도서 표지 이미지 경로를 저장할 리스트 생성
    List<String> bookCoverPaths = new ArrayList<>();
    // Gson 객체 생성
    Gson gson = new Gson();

    // 사용자가 로그인되어 있는지 확인
    if (userId == null) {
        // 사용자가 로그인되어 있지 않으면 401 Unauthorized 응답 반환
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.getWriter().write("Unauthorized Access");
    } else {
        try (
            // 데이터베이스 연결
            Connection con = DatabaseConnector.getConnection();
            // 사용자의 도서와 표지 이미지 정보를 가져오는 쿼리 준비
            PreparedStatement pstmt = con.prepareStatement("SELECT b.book_name, i.cover_image FROM book b JOIN image i ON b.book_id = i.book_id WHERE b.user_id = ?");
        ) {
            // 쿼리 매개변수 설정
            pstmt.setInt(1, userId);
            // 쿼리 실행 결과 가져오기
            ResultSet rs = pstmt.executeQuery();
            // 결과 순회하면서 도서 표지 이미지 경로 생성 및 리스트에 추가
            while (rs.next()) {
                String bookName = rs.getString("book_name");
                String coverImage = rs.getString("cover_image");
                String imagePath = "image/" + bookName + "/" + coverImage;
                bookCoverPaths.add(imagePath);
                // 콘솔에 로그 남기기
                System.out.println("Generated image path: " + imagePath);
            }
        } catch (SQLException e) {
            e.printStackTrace();
            // 데이터베이스 오류 처리
        }
    }

    // 도서 표지 이미지 경로를 JSON 형식으로 변환
    String json = gson.toJson(bookCoverPaths);
    // 클라이언트로 JSON 응답 반환
    response.getWriter().write(json);
%>

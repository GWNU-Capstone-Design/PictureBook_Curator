const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');
const multer = require('multer');
const bcrypt = require('bcrypt');
const path = require('path');
const session = require('express-session');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const saltRounds = 10;

const app = express();
const port = 3000;

// 파일 저장을 위한 multer 설정 변경
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage: storage });
const multipleUpload = upload.fields([
  { name: 'bookCover', maxCount: 1 },
  { name: 'bookContent', maxCount: 10 },
]);

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));
app.use('/uploads', express.static('uploads'));
// 오디오 사용을 위한 정적 파일 제공 설정
app.use('/static', express.static(path.join(__dirname, 'static')));


app.use(
  session({
    secret: '1234', 
    resave: false,
    saveUninitialized: true,
    cookie: { secure: !true },
  })
);

const connection = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: '1234',
  database: 'database',
});

connection.connect((err) => {
  if (err) {
    console.error('error connecting: ' + err.stack);
    return;
  }
  console.log('Connected to the MySQL server.');
});

// 로그인 페이지로 리다이렉트
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'LogIn.html'));
});

// 로그인 처리
app.post('/login', (req, res) => {
  const { loginEmail, loginPassword } = req.body;
  connection.query(
    'SELECT * FROM user WHERE user_email = ?',
    [loginEmail],
    (error, results) => {
      if (results.length > 0) {
        bcrypt.compare(
          loginPassword,
          results[0].user_pw,
          (err, result) => {
            if (result) {
              req.session.userId = results[0].user_id;
              res.redirect('/main.html');
            } else {
              res.send('Login failed');
            }
          }
        );
      } else {
        res.send('User not found');
      }
    }
  );
});

// 회원가입 처리
app.post('/signup', (req, res) => {
  const { signupEmail, signupName, signupPassword } = req.body;
  bcrypt.hash(signupPassword, saltRounds, (err, hash) => {
    if (err) {
      return res.status(500).send('Error hashing password');
    }
    const query =
      'INSERT INTO user (user_email, user_name, user_pw) VALUES (?, ?, ?)';
    connection.query(query, [signupEmail, signupName, hash], (error, results) => {
      if (error) {
        console.error(error);
        return res.status(500).send('Database error');
      }
      res.redirect('/LogIn.html');
    });
  });
});

// 사용자의 책 표지 정보만 불러오는 API
app.get('/getUserBookCovers', (req, res) => {
  if (!req.session.userId) {
    return res.status(401).send('Not logged in');
  }

  const query = `
    SELECT b.book_id, b.book_name, i.image_value
    FROM book b
    JOIN image i ON b.book_id = i.book_id
    WHERE b.user_id = ? AND i.is_cover = 1;  -- is_cover 컬럼을 사용해 표지 이미지만 조회
  `;
  connection.query(query, [req.session.userId], (error, results) => {
    if (error) {
      console.error('Database error:', error);
      return res.status(500).send('Database error');
    }
    res.json(results);
  });
});

app.get('/getBookDetails/:bookId', (req, res) => {
  const bookId = req.params.bookId;
  const query = `
      SELECT b.book_name, i.image_value, t.text_value
      FROM book b
      JOIN image i ON b.book_id = i.book_id
      LEFT JOIN text t ON i.image_id = t.image_id
      WHERE b.book_id = ? AND i.is_cover = 0; -- 표지 이미지를 제외
  `;
  connection.query(query, [bookId], (error, results) => {
      if (error) {
          console.error('Database error:', error);
          return res.status(500).send({ message: 'Database error', error: error });
      }
      console.log('Results from DB:', results);
      res.json({ 
          bookName: results.length > 0 ? results[0].book_name : '',
          images: results.map(item => ({
              imageUrl: item.image_value,
              text: item.text_value && typeof item.text_value === 'object' ? item.text_value.text : ''
          }))
      });
  });
});




// 책 추가 처리 라우트 변경
app.post('/addBook', multipleUpload, async (req, res) => {
  const { bookName } = req.body;
  const bookCover = req.files['bookCover'][0].path;  // 표지 파일 경로
  const bookContents = req.files['bookContent'] ? req.files['bookContent'].map(file => file.path) : [];  // 내용 파일 경로 배열

  const user_id = req.session.userId;

  // 책 기본 정보를 먼저 데이터베이스에 추가
  connection.query('INSERT INTO book (user_id, book_name) VALUES (?, ?)', [user_id, bookName], (error, results) => {
      if (error) {
          console.error('Error inserting new book:', error);
          return res.status(500).send('Database error when inserting book');
      }
      const newBookId = results.insertId;

      // 표지 이미지 데이터베이스에 추가
      const coverInsertQuery = 'INSERT INTO image (book_id, image_value, is_cover) VALUES (?, ?, 1)';
      connection.query(coverInsertQuery, [newBookId, bookCover], (coverError) => {
          if (coverError) {
              console.error('Error inserting book cover:', coverError);
          }
      });

      // 내용 이미지 데이터베이스에 추가
      bookContents.forEach(content => {
          const contentInsertQuery = 'INSERT INTO image (book_id, image_value, is_cover) VALUES (?, ?, 0)';
          connection.query(contentInsertQuery, [newBookId, content], (contentError) => {
              if (contentError) {
                  console.error('Error inserting book content image:', contentError);
              }
          });
      });

      // 이미지들을 jy.zvz.be:5000/process_images로 전송하고 결과를 처리
      const form = new FormData();
      bookContents.forEach(content => {
          form.append('images', fs.createReadStream(content));
      });

      console.log('Sending images to server:', bookContents);

      axios.post('http://jy.zvz.be:5000/process_images', form, { headers: form.getHeaders() })
          .then(response => {
              console.log('Server response:', response.data);
              const gptResults = response.data.gpt_results;
              const gptAudioUrls = response.data.gpt_audio_urls;

              // GPT 텍스트와 오디오 URL을 데이터베이스에 저장
              gptResults.forEach((text, index) => {
                  const imageIdQuery = 'SELECT image_id FROM image WHERE book_id = ? AND image_value = ? AND is_cover = 0';
                  connection.query(imageIdQuery, [newBookId, bookContents[index]], (imageError, imageResults) => {
                      if (imageError) {
                          console.error('Error retrieving image ID:', imageError);
                      } else if (imageResults.length > 0) {
                          const imageId = imageResults[0].image_id;
                          const textInsertQuery = 'INSERT INTO text (image_id, text_value) VALUES (?, ?)';
                          connection.query(textInsertQuery, [imageId, JSON.stringify({ text })], (textError) => {
                              if (textError) {
                                  console.error('Error inserting GPT text:', textError);
                              }
                          });
                      }
                  });
              });
              res.redirect('/main.html');
          })
          .catch(error => {
              console.error('Error processing images with GPT:', error);
              if (error.response) {
                  console.error('Response data:', error.response.data);
                  console.error('Response status:', error.response.status);
                  console.error('Response headers:', error.response.headers);
              } else if (error.request) {
                  console.error('Request data:', error.request);
              } else {
                  console.error('Error message:', error.message);
              }
              res.status(500).send('Error processing images with GPT');
          });
  });
});



app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
